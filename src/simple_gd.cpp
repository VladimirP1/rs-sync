#include <iomanip>
#include <iostream>
#include <vector>

#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "cv.hpp"

#include <telemetry-parser.h>

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
    nproblem.each_row([](arma::mat& m) { m /= arma::norm(m); });

    arma::vec3 best_sol;
    double least_med = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 200; ++i) {
        int vs[3];
        vs[0] = vs[1] = rand() % problem.n_rows;
        while (vs[1] == vs[0]) vs[2] = vs[1] = rand() % problem.n_rows;
        while (vs[2] == vs[1] || vs[2] == vs[0]) vs[2] = rand() % problem.n_rows;

        arma::mat v = arma::trans(arma::normalise(arma::cross(
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

arma::vec3 opt_guess_translational_motion2(const arma::mat& problem) {
    arma::vec3 best_sol;
    static constexpr double k = 1e3;
    double least_med = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 200; ++i) {
        int vs[3];
        vs[0] = vs[1] = rand() % problem.n_rows;
        while (vs[1] == vs[0]) vs[2] = vs[1] = rand() % problem.n_rows;
        while (vs[2] == vs[1] || vs[2] == vs[0]) vs[2] = rand() % problem.n_rows;

        arma::mat v = arma::trans(arma::normalise(arma::cross(
            problem.row(vs[0]) - problem.row(vs[1]), problem.row(vs[0]) - problem.row(vs[2]))));

        arma::mat r = (problem * v) * k;
        arma::mat rho = arma::log1p(r % r);

        double med = arma::accu(rho);
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

        opt_compute_problem(frame_, gyro_delay[0], *optdata_, P);
        opt_compute_problem(frame_, gyro_delay[0] - kStep, *optdata_, PL);
        opt_compute_problem(frame_, gyro_delay[0] + kStep, *optdata_, PR);

        double r1 = calc(PL, M, k);
        double r2 = calc(PR, M, k);

        auto [v1, j1] = std::make_tuple(P * M, P);
        auto [v2, j2] = sqr_jac(v1);

        auto [v3, j3] = sqr_jac(M);
        auto [v4, j4] = sum_jac(v3);
        auto [v5, j5, _] = div_jac(v4, k * k);

        auto [v6, j6a, j6b] = div_jac(v2, v5[0]);
        auto [v7, j7] = log1p_jac(v6);
        auto [v8, j8] = sqrt_jac(v7);
        auto [v9, j9] = sum_jac(v8);

        cost = v9;

        jac_gyro_delay = (r2 - r1) / 2 / kStep;

        jac_M = j9 * j8 * j7 * (j6a * j2 * j1 + j6b * j5 * j4 * j3);

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
        arma::mat r = (P * M) * (k / arma::norm(M));
        arma::mat rho = arma::log1p(r % r);
        return arma::accu(arma::sqrt(rho));
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
        // std::cout << v << " " << v1 << std::endl;
        if (v - v1 >= t * hyper.c * m) break;
        t *= hyper.tau;
        // if (i == limit - 1) std::cout << "armijo fail" << std::endl;
    }
    // std::cout << t << std::endl;
    return -t * p;
}

struct opt_result {
    double delay;
    double cost;
};

opt_result opt_run(OptData& data, double initial_delay) {
    static constexpr backtrack_hyper motion_hyper = {
        .c = 1e-4, .tau = .1, .limit = 20, .step_init = 1e-2};

    static constexpr backtrack_hyper delay_hyper = {
        .c = .1e-4, .tau = .1, .limit = 10, .step_init = 1};

    std::vector<std::unique_ptr<FrameState>> costs;
    for (auto& [frame, _] : data.flows) {
        costs.push_back(std::make_unique<FrameState>(frame, &data));
        costs.back()->motion_vec = costs.back()->GuessMotion(initial_delay);
        costs.back()->opt_tmp_data.resize(3, 1);
        costs.back()->opt_tmp_data.zeros();
        costs.back()->gyro_delay = initial_delay;
    }

    for (int i = 0; i < 1000; ++i) {
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

            auto del_f = [&](arma::vec x) {
                arma::mat cost, del_jac, mot_jac;
                fs->Cost(x, fs->motion_vec, cost, del_jac, mot_jac);
                return std::make_pair(cost[0], arma::vec{del_jac});
            };
            auto del_f_only = [&](arma::vec x) {
                arma::mat cost;
                fs->CostOnly(x, fs->motion_vec, cost);
                return cost[0];
            };

            for (int i = 0; i < 50; ++i) {
                auto bt = backtrack(f, f_only, fs->motion_vec, motion_hyper);
                fs->motion_vec += bt;
                if (arma::norm(bt) < 1e-6) break;
            }

            arma::mat bt2 = backtrack(del_f, del_f_only, fs->gyro_delay, delay_hyper);

            fs->gyro_delay += bt2;
            std::cout << fs->gyro_delay[0] << " ";

            if (rand() % 1000 < 5) fs->motion_vec = costs.back()->GuessMotion(fs->gyro_delay[0]);
        }
        std::cout << std::endl;
    }

    return {0, 0};
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
    track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 90 + 30);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 400, 430);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1300, 1330);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1400, 1430);
    double delay = -44.7;
    for (int i = 0; i < 4; ++i) delay = opt_run(opt_data, delay).delay;
    // plot_run(opt_data);
}
