#include <execution>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#include "cv.hpp"
#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "backtrack.hpp"
#include "simple_calculus.hpp"
#include "inline_utils.hpp"
#include "gsl.hpp"
#include "loss.hpp"
#include "io.hpp"

struct FrameState {
    FrameState(int frame, OptData* optdata) : frame_{frame}, optdata_{optdata} {}

    bool Cost(const arma::mat& gyro_delay, const arma::mat& M, arma::mat& cost,
              arma::mat& jac_gyro_delay, arma::mat& jac_M) const {
        arma::mat P, PL, PR;

        opt_compute_problem(frame_, gyro_delay[0], *optdata_, P);
        opt_compute_problem(frame_, gyro_delay[0] - kStep, *optdata_, PL);
        opt_compute_problem(frame_, gyro_delay[0] + kStep, *optdata_, PR);

        double r1 = calc(PL, M, var_k);
        double r2 = calc(PR, M, var_k);

        auto [v1, j1] = std::make_tuple(P * M, P);
        auto [v2, j2] = sqr_jac(v1);

        auto [v3, j3] = sqr_jac(M);
        auto [v4, j4] = sum_jac(v3);
        auto [v5, j5, _] = div_jac(v4, var_k * var_k);

        auto [v6, j6a, j6b] = div_jac(v2, v5[0]);
        auto [v7, j7] = log1p_jac(v6);
        auto [v8, j8] = sum_jac(v7);

        cost = v8;

        jac_gyro_delay = (r2 - r1) / 2 / kStep;

        jac_M = j8 * j7 * (j6a * j2 * j1 + j6b * j5 * j4 * j3);

        return true;
    }

    bool CostOnly(const arma::mat& gyro_delay, const arma::mat& M, arma::mat& cost) const {
        arma::mat P;
        opt_compute_problem(frame_, gyro_delay[0], *optdata_, P);
        cost = {calc(P, M, var_k)};
        return true;
    }

    arma::vec3 GuessMotion(double gyro_delay) const {
        arma::mat problem;
        opt_compute_problem(frame_, gyro_delay, *optdata_, problem);
        return opt_guess_translational_motion(problem);
    }

    double GuessK(double gyro_delay) const {
        arma::mat problem;
        opt_compute_problem(frame_, gyro_delay, *optdata_, problem);
        return 1 / arma::norm(problem * motion_vec) * 1e2;
    }

    arma::mat gyro_delay;

    arma::vec3 motion_vec;
    double var_k = 1e3;
    arma::mat opt_tmp_data;

   private:
    static constexpr double kStep = 1e-6;

    int frame_;
    OptData* optdata_;

    static double calc(arma::mat P, arma::mat M, double k) {
        arma::mat r = (P * M) * (k / arma::norm(M));
        arma::mat rho = arma::log1p(r % r);
        return arma::accu(rho);
    }
};

double opt_run(OptData& data, double initial_delay, int min_frame = std::numeric_limits<int>::min(),
               int max_frame = std::numeric_limits<int>::max()) {
    arma::mat gyro_delay(1, 1);
    gyro_delay[0] = initial_delay;

    std::vector<std::unique_ptr<FrameState>> costs;
    for (auto& [frame, _] : data.flows.data) {
        if (frame < min_frame || frame > max_frame) continue;
        costs.push_back(std::make_unique<FrameState>(frame, &data));
        costs.back()->motion_vec = costs.back()->GuessMotion(gyro_delay[0]);
        costs.back()->var_k = costs.back()->GuessK(gyro_delay[0]);
        costs.back()->opt_tmp_data.resize(3, 1);
        costs.back()->opt_tmp_data.zeros();
    }

    Backtrack delay_optimizer;
    delay_optimizer.SetHyper(.2, .1, 1, 10);

    delay_optimizer.SetObjective([&](arma::vec x) {
        std::mutex m;
        arma::mat cost(1, 1), delay_g(1, 1);
        std::for_each(std::execution::par, costs.begin(), costs.end(),
                      [x, &cost, &delay_g, &m](auto& fs) {
                          arma::mat cur_cost, cur_delay_g, tmp;
                          fs->Cost(x, fs->motion_vec, cur_cost, cur_delay_g, tmp);
                          std::unique_lock<std::mutex> lock(m);
                          cost += cur_cost;
                          delay_g += cur_delay_g;
                      });
        return std::make_pair(cost[0], delay_g);
    });

    delay_optimizer.SetObjective([&](arma::vec x) {
        std::mutex m;
        arma::mat cost(1, 1);
        std::for_each(std::execution::par, costs.begin(), costs.end(), [x, &cost, &m](auto& fs) {
            arma::mat cur_cost;
            fs->CostOnly(x, fs->motion_vec, cur_cost);
            std::unique_lock<std::mutex> lock(m);
            cost += cur_cost;
        });
        return cost[0];
    });

    struct delay_opt_info {
        double step_size;
    };

    constexpr double delay_b{.3};
    arma::mat delay_v(1, 1);
    auto do_opt_motion = [&]() {
        std::for_each(std::execution::par, costs.begin(), costs.end(), [gyro_delay](auto& fs) {
            Gsl::MultiminFunction func(3);
            func.SetF([&fs, gyro_delay](arma::vec x) {
                arma::mat cost;
                fs->CostOnly(gyro_delay, x, cost);
                return cost[0];
            });
            func.SetFdF([&fs, gyro_delay](arma::vec x) {
                arma::mat cost, del_jac, mot_jac;
                fs->Cost(gyro_delay, x, cost, del_jac, mot_jac);
                return std::make_pair(cost[0], arma::vec{mot_jac.t()});
            });
            gsl_vector* motion_vec = gsl_vector_alloc(3);

            gsl_multimin_fdfminimizer* minimizer =
                gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_vector_bfgs2, 3);

            arma::wrap(motion_vec) = fs->motion_vec;
            gsl_multimin_fdfminimizer_set(minimizer, &func.gsl_func, motion_vec, 1e-2, 1e-2);

            for (int j = 0; j < 500; ++j) {
                auto r = gsl_multimin_fdfminimizer_iterate(minimizer);
                if (r != GSL_SUCCESS) {
                    break;
                }

                if (gsl_multimin_test_gradient(minimizer->gradient, 1e-6) == GSL_SUCCESS) {
                    break;
                }
            }
            fs->motion_vec = arma::wrap(minimizer->x);
            gsl_vector_free(motion_vec);
            gsl_multimin_fdfminimizer_free(minimizer);
        });
    };

    auto do_opt_delay = [&]() {
        arma::mat step = delay_optimizer.Step(gyro_delay - delay_b * delay_v);

        delay_v = delay_b * delay_v + step;
        gyro_delay += delay_v;

        return delay_opt_info{arma::norm(step)};
    };

    int converge_counter = 0;

    for (int i = 0; i < 1000; i++) {
        // Optimize motion
        do_opt_motion();

        // Optimize delay
        auto info = do_opt_delay();

        if (info.step_size < 1e-6) {
            converge_counter++;
        } else {
            converge_counter = 0;
        }

        if (converge_counter > 5) {
            break;
        }

        std::cerr << gyro_delay[0] << " " << info.step_size << std::endl;
    }

    return gyro_delay[0];
}

void plot_run(OptData& data) {
    Backtrack motion_optimizer;
    motion_optimizer.SetHyper(1e-4, .1, 1e-2, 20);

    for (double pos = -60; pos < -30; pos += .1) {
        std::vector<std::unique_ptr<FrameState>> costs;
        for (auto& [frame, _] : data.flows.data) {
            costs.push_back(std::make_unique<FrameState>(frame, &data));
            costs.back()->motion_vec = costs.back()->GuessMotion(pos);
            costs.back()->opt_tmp_data.resize(3, 1);
            costs.back()->opt_tmp_data.zeros();
            costs.back()->gyro_delay = pos;
        }

        for (auto& fs : costs) {
            motion_optimizer.SetObjective([&](arma::vec x) {
                arma::mat cost, del_jac, mot_jac;
                fs->Cost(fs->gyro_delay, x, cost, del_jac, mot_jac);
                return std::make_pair(cost[0], arma::vec{mot_jac.t()});
            });
            motion_optimizer.SetObjective([&](arma::vec x) {
                arma::mat cost;
                fs->CostOnly(fs->gyro_delay, x, cost);
                return cost[0];
            });

            for (int i = 0; i < 500; ++i) {
                auto bt = motion_optimizer.Step(fs->motion_vec);
                fs->motion_vec += bt;
                if (arma::norm(bt) < 1e-6) break;
            }
        }

        auto total_cost = [&]() {
            arma::mat cost(1, 1), delay_g(1, 1);
            for (auto& fs : costs) {
                arma::mat cur_cost;
                fs->CostOnly(fs->gyro_delay, fs->motion_vec, cur_cost);
                cost += cur_cost;
            }
            return cost[0];
        };
        std::cout << pos << "," << total_cost() << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(14);

    OptData opt_data;
    // YXZ yZX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "yZX");
    // optdata_fill_gyro(opt_data, "GH011230.MP4", "yZX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    // track_frames(opt_data.flows, lens, "GH011230.MP4", 90, 1000);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 90 + 30);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 600, 630);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1700, 1710);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 250);
    track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 1750);
    // double delay = -44.7;
    // for (int i = 0; i < 4; ++i) delay = opt_run(opt_data, delay).delay;

    for (int pos = 90; pos < 1600; pos += 60) {
        std::cerr << pos << std::endl;
        double delay = -42;
        for (int i = 0; i < 4; ++i) delay = opt_run(opt_data, delay, pos, pos + 150);
        std::cout << pos << "," << delay << std::endl;
    }

    // plot_run(opt_data);
}
