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
    gyro_delay /= 1000;
    const auto& flow = data.flows.at(frame);

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
        auto [v8, j8] = sum_jac(v7);

        cost = v8;

        jac_gyro_delay = (r2 - r1) / 2 / kStep;

        jac_M = j8 * j7 * (j6a * j2 * j1 + j6b * j5 * j4 * j3);

        return true;
    }

    arma::vec3 GuessMotion(double gyro_delay) {
        arma::mat problem;
        opt_compute_problem(frame_, gyro_delay, *optdata_, problem);
        return opt_guess_translational_motion(problem);
    }

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
        return arma::accu(rho);
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
};

static arma::vec backtrack(std::function<std::tuple<double, arma::vec>(arma::vec)> f,
                           arma::vec x0) {
    static constexpr double c = .7;
    static constexpr double tau = .1;
    static constexpr int limit = 20;

    auto [v, p] = f(x0);
    double m = arma::dot(p, p);
    double t = 1;
    for (int i = 0; i < limit; ++i) {
        auto [v1, m1] = f(x0 - t * p);
        // std::cout << v << " " << v1 << std::endl;
        if (v - v1 >= t * c * m) break;
        t *= tau;
        // if (i == limit - 1) std::cout << "armijo fail" << std::endl;
    }
    // std::cout << t << std::endl;
    return x0 - t * p;
}

void opt_run(OptData& data) {
    arma::mat gyro_delay(1, 1);
    gyro_delay[0] = -47;

    std::vector<std::unique_ptr<FrameState>> costs;
    for (auto& [frame, _] : data.flows) {
        costs.push_back(std::make_unique<FrameState>(frame, &data));
        costs.back()->motion_vec = costs.back()->GuessMotion(gyro_delay[0]);
        costs.back()->opt_tmp_data.resize(3, 1);
        costs.back()->opt_tmp_data.zeros();
    }

    constexpr double delay_lr = 1e-3;
    constexpr double motion_lr = 1e-4;

    for (int i = 0; i < 9000; i++) {
        // Optimize motion
        for (auto& fs : costs) {
            for (int j = 0; j < 1; ++j) {
                fs->motion_vec = backtrack(
                    [&](arma::vec x) {
                        arma::mat cost, del_jac, mot_jac;
                        fs->Cost(gyro_delay, x, cost, del_jac, mot_jac);
                        return std::make_pair(cost[0], arma::vec{mot_jac.t()});
                    },
                    fs->motion_vec);
            }
        }

        // Optimize delay
        gyro_delay = backtrack(
            [&](arma::vec x) {
                arma::mat cost(1, 1), delay_g(1, 1);
                for (auto& fs : costs) {
                    arma::mat cur_cost, cur_delay_g, tmp;
                    fs->Cost(x, fs->motion_vec, cur_cost, cur_delay_g, tmp);
                    cost += cur_cost;
                    delay_g += cur_delay_g;
                }
                return std::make_pair(cost[0], delay_g);
            },
            gyro_delay);


        // if (i%100 == 0)
        std::cout << gyro_delay[0] << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(16);

    OptData opt_data;
    // YXZ zYX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "zYX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 300, 330);
    track_frames(opt_data.flows, lens, "GX011338.MP4", 400, 450);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1300, 1330);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1400, 1430);

    opt_run(opt_data);
}
