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

#include <telemetry-parser.h>

arma::mat safe_normalize(arma::mat m) {
    double norm = arma::norm(m);
    if (norm < 1e-12) {
        return m;
    }
    return m / norm;
}

int mtrand(const int& min, const int& max) {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

struct OptData {
    double quats_start{};
    int sample_rate{};
    ndspline quats{};

    FramesFlow flows{};
};

void optdata_fill_gyro(OptData& optdata, const char* filename, const char* orient) {
    tp_gyrodata data = tp_load_gyro(filename, orient);
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    tp_free(data);
    optdata.sample_rate = gyro_interpolate(timestamps, gyro);

    arma::mat quats(4, data.samples);
    quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < quats.n_cols; ++i) {
        quats.col(i) = arma::normalise(
            quat_prod(quats.col(i - 1), quat_from_aa(gyro.col(i) / optdata.sample_rate)));
    }
    optdata.quats_start = timestamps.front();
    optdata.quats = ndspline::make(quats);
}

void opt_compute_problem(int frame, double gyro_delay, const OptData& data, arma::mat& problem) {
    const auto& flow = data.flows.at(frame);
    gyro_delay /= 1000;

    arma::mat ap = flow.rows(0, 2);
    arma::mat bp = flow.rows(3, 5);
    arma::mat at = (flow.row(6) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat bt = (flow.row(7) - data.quats_start + gyro_delay) * data.sample_rate;

    problem.resize(at.n_cols, 3);
    for (int i = 0; i < at.n_cols; ++i) {
        arma::vec4 a_conj = quat_conj(data.quats.eval(at[i]));
        arma::vec4 b = data.quats.eval(bt[i]);
        double inv_ab_norm = (1. / arma::norm(a_conj)) * (1. / arma::norm(b));
        arma::vec4 rot = quat_prod(a_conj, b) * inv_ab_norm;
        arma::vec3 br = quat_rotate_point(quat_conj(rot), bp.col(i));
        problem.row(i) = arma::trans(arma::cross(ap.col(i), br));
    }
}

arma::vec3 opt_guess_translational_motion(const arma::mat& problem) {
    arma::mat nproblem = problem;
    nproblem.each_row([](arma::mat& m) { m = safe_normalize(m); });

    arma::vec3 best_sol;
    double least_med = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 200; ++i) {
        int vs[3];
        vs[0] = vs[1] = mtrand(0, problem.n_rows - 1);
        while (vs[1] == vs[0]) vs[2] = vs[1] = mtrand(0, problem.n_rows - 1);
        while (vs[2] == vs[1] || vs[2] == vs[0]) vs[2] = mtrand(0, problem.n_rows - 1);

        arma::mat v = arma::trans(safe_normalize(arma::cross(
            problem.row(vs[0]) - problem.row(vs[1]), problem.row(vs[0]) - problem.row(vs[2]))));

        arma::mat residuals = nproblem * v;
        arma::mat residuals2 = residuals % residuals;

        std::sort(residuals2.begin(), residuals2.end());
        double med = residuals2(residuals2.n_rows / 4, 0);
        if (med < least_med) {
            least_med = med;
            best_sol = v;
        }
    }
    return best_sol;
}

struct FrameState {
    FrameState(int frame, OptData* optdata) : frame_{frame}, optdata_{optdata} {}

    bool Cost(const arma::mat& gyro_delay, const arma::mat& M, arma::mat& cost,
              arma::mat& jac_gyro_delay, arma::mat& jac_M) const {
        arma::mat P, PL, PR;

        opt_compute_problem(frame_, gyro_delay[0] - kStep, *optdata_, PL);
        opt_compute_problem(frame_, gyro_delay[0] + kStep, *optdata_, PR);

        double r1 = calc(PL, M, k);
        double r2 = calc(PR, M, k);

        cost = (r1 + r2) / 2;

        jac_gyro_delay = (r2 - r1) / 2 / kStep;

        jac_M.resize(1, 3);
        jac_M.zeros();

        return true;
    }

    bool CostOnly(const arma::mat& gyro_delay, const arma::mat& M, arma::mat& cost) const {
        arma::mat P;
        opt_compute_problem(frame_, gyro_delay[0], *optdata_, P);
        cost = {calc(P, M, k)};
        return true;
    }

    arma::vec3 GuessMotion(double gyro_delay) {
        arma::mat problem;
        opt_compute_problem(frame_, gyro_delay, *optdata_, problem);
        return opt_guess_translational_motion(problem);
    }

    arma::mat gyro_delay;

    arma::vec3 motion_vec;
    arma::mat opt_tmp_data;

   private:
    static constexpr double k = 1e3;
    static constexpr double kStep = 1e-6;

    int frame_;
    OptData* optdata_;

    static double calc(arma::mat P, arma::mat M, double k) {
        // P.each_row([k](arma::mat& row) { row = row / (1. + arma::norm(row)); });
        // double res = 0;
        // for (int i = 0; i < P.n_rows; ++i) {
        //     res += arma::sum(P.row(i) % P.row(i));
        // }
        arma::mat rho = arma::log1p(arma::sum(P % P, 1) * k * k);
        // std::cerr << (arma::sum(P % P, 1) * k * k).t() << std::endl;
        return arma::accu(rho);
        // return res;
    }

    static std::tuple<arma::mat, arma::mat> sqr_jac(arma::mat x) {
        return {x % x, arma::diagmat(2. * x)};
    }

    static std::tuple<arma::mat, arma::mat> sqrt_jac(arma::mat x) {
        x = arma::sqrt(x);
        return {x, arma::diagmat(1. / (2. * x))};
    }

    static std::tuple<arma::mat, arma::mat> log1p_jac(arma::mat x) {
        return {arma::log1p(x), arma::diagmat(1. / (1. + x))};
    }

    static std::tuple<arma::mat, arma::mat> sum_jac(arma::mat x) {
        arma::mat d(1, x.n_rows);
        d.ones();
        return {arma::sum(x), d};
    }

    static std::tuple<arma::mat, arma::mat, arma::mat> div_jac(arma::mat x, double y) {
        arma::mat dx(x.n_rows, x.n_rows);
        dx.eye();
        return {x / y, dx / y, -x / (y * y)};
    }

    static std::tuple<arma::mat, arma::mat> mul_const_jac(arma::mat x, double y) {
        arma::mat dx(x.n_rows, x.n_rows);
        dx.eye();
        return {x * y, dx * y};
    }
};

struct backtrack_hyper {
    double c{.7};
    double tau{.1};
    double limit{20};
    double step_init{1};
};

static arma::vec backtrack(std::function<std::tuple<double, arma::vec>(arma::vec)> f,
                           std::function<double(arma::vec)> f_vonly, arma::vec x0,
                           backtrack_hyper hyper = {}) {
    if (!f_vonly) f_vonly = [&](arma::vec x) { return std::get<0>(f(x)); };

    auto [v, p] = f(x0);
    double m = arma::dot(p, p);
    double t = hyper.step_init;
    for (int i = 0; i < hyper.limit; ++i) {
        auto v1 = f_vonly(x0 - t * p);
        if (v - v1 >= t * hyper.c * m) break;
        t *= hyper.tau;
    }
    return -t * p;
}

struct opt_result {
    double delay;
    double cost;
};

opt_result opt_run(OptData& data, double initial_delay,
                   int min_frame = std::numeric_limits<int>::min(),
                   int max_frame = std::numeric_limits<int>::max()) {
    arma::mat gyro_delay(1, 1);
    gyro_delay[0] = initial_delay;

    static constexpr backtrack_hyper motion_hyper = {
        .c = .7, .tau = .1, .limit = 20, .step_init = 1e-2};

    static constexpr backtrack_hyper delay_hyper = {
        .c = .2, .tau = .1, .limit = 10, .step_init = 1};

    std::vector<std::unique_ptr<FrameState>> costs;
    for (auto& [frame, _] : data.flows) {
        if (frame < min_frame || frame > max_frame) continue;
        costs.push_back(std::make_unique<FrameState>(frame, &data));
        costs.back()->motion_vec = costs.back()->GuessMotion(gyro_delay[0]);
        costs.back()->opt_tmp_data.resize(3, 1);
        costs.back()->opt_tmp_data.zeros();
    }

    struct delay_opt_info {
        double step_size;
    };

    constexpr double delay_b{.3};
    arma::mat delay_v(1, 1);
    auto do_opt_motion = [&]() {
        std::for_each(std::execution::par, costs.begin(), costs.end(), [gyro_delay](auto& fs) {
            for (int j = 0; j < 500; ++j) {
                auto bt = backtrack(
                    [&fs, gyro_delay](arma::vec x) {
                        arma::mat cost, del_jac, mot_jac;
                        fs->Cost(gyro_delay, x, cost, del_jac, mot_jac);
                        return std::make_pair(cost[0], arma::vec{mot_jac.t()});
                    },
                    [&fs, gyro_delay](arma::vec x) {
                        arma::mat cost;
                        fs->CostOnly(gyro_delay, x, cost);
                        return cost[0];
                    },
                    fs->motion_vec, motion_hyper);
                fs->motion_vec += bt;
                if (arma::norm(bt) < 1e-6) {
                    break;
                }
            }
        });
    };

    auto f = [&](arma::vec x) {
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
    };

    auto f_only = [&](arma::vec x) {
        std::mutex m;
        arma::mat cost(1, 1);
        std::for_each(std::execution::par, costs.begin(), costs.end(), [x, &cost, &m](auto& fs) {
            arma::mat cur_cost;
            fs->CostOnly(x, fs->motion_vec, cur_cost);
            std::unique_lock<std::mutex> lock(m);
            cost += cur_cost;
        });
        return cost[0];
    };

    auto do_opt_delay = [&]() {
        arma::mat bt = backtrack(f, f_only, gyro_delay - delay_b * delay_v, delay_hyper);

        delay_v = delay_b * delay_v + bt;
        gyro_delay += delay_v;

        return delay_opt_info{arma::norm(bt)};
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
            // std::cout << "Converged at " << i << std::endl;
            break;
        }

        std::cerr << gyro_delay[0] << " " << f_only(gyro_delay) << " " << info.step_size
                  << std::endl;
    }

    return {gyro_delay[0], f_only(gyro_delay)};
}

void plot_run(OptData& data) {
    static constexpr backtrack_hyper motion_hyper = {
        .c = .01, .tau = .1, .limit = 20, .step_init = 1e-2};

    for (double pos = -60; pos < -30; pos += .1) {
        std::vector<std::unique_ptr<FrameState>> costs;
        for (auto& [frame, _] : data.flows) {
            costs.push_back(std::make_unique<FrameState>(frame, &data));
            costs.back()->motion_vec = costs.back()->GuessMotion(pos);
            costs.back()->opt_tmp_data.resize(3, 1);
            costs.back()->opt_tmp_data.zeros();
            costs.back()->gyro_delay = pos;
        }

        for (auto& fs : costs) {
            auto f = [&](arma::vec x) {
                arma::mat cost, del_jac, mot_jac;
                fs->Cost(fs->gyro_delay, x, cost, del_jac, mot_jac);
                return std::make_pair(cost[0], arma::vec{mot_jac.t()});
            };
            auto f_only = [&](arma::vec x) {
                arma::mat cost;
                fs->CostOnly(fs->gyro_delay, x, cost);
                return cost[0];
            };

            for (int i = 0; i < 500; ++i) {
                auto bt = backtrack(f, f_only, fs->motion_vec, motion_hyper);
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
    std::cout << std::fixed << std::setprecision(3);

    OptData opt_data;
    // YXZ yZX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "yZX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 90 + 30);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 600, 630);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1700, 1710);
    track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 1750);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 90+150);
    // double delay = -44.7;
    // for (int i = 0; i < 4; ++i) delay = opt_run(opt_data, delay).delay;

    for (int pos = 90; pos < 1600; pos += 60) {
        std::cerr << pos << std::endl;
        double delay = -42;
        for (int i = 0; i < 4; ++i) delay = opt_run(opt_data, delay, pos, pos + 150).delay;
        std::cout << pos << "," << delay << std::endl;
    }

    // plot_run(opt_data);
}
