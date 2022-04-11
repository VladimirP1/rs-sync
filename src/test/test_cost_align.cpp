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

#include <telemetry-parser.h>

struct OptData {
    double quats_start{};
    int sample_rate{};
    ndspline quats{};

    FramesFlow flows{};
};

struct params {
    arma::vec delay() { return arma::vec(params.memptr(), 1, false, true); }
    arma::vec motion() { return arma::vec(params.memptr() + 1, 1, false, true); }
    arma::vec align() { return arma::vec(params.memptr() + 4, 1, false, true); }
    const arma::vec delay() const { return arma::vec(params.memptr(), 1, false, true); }
    const arma::vec motion() const { return arma::vec(params.memptr() + 1, 1, false, true); }
    const arma::vec align() const { return arma::vec(params.memptr() + 4, 1, false, true); }
    mutable arma::vec params{7};
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

arma::mat opt_compute_rotations(int frame, const params& p, const OptData& data) {
    const auto& flow = data.flows.data.at(frame);
    double gyro_delay = p.delay()[0] / 1000;
    arma::mat at = (flow.row(6) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat bt = (flow.row(7) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat rotations(flow.n_cols, 4);
    for (int i = 0; i < flow.n_cols; ++i) {
        arma::vec4 a_conj = quat_conj(data.quats.eval(at[i]));
        arma::vec4 b = data.quats.eval(bt[i]);
        double inv_ab_norm = (1. / arma::norm(a_conj)) * (1. / arma::norm(b));
        arma::vec4 rot = quat_prod(a_conj, b) * inv_ab_norm;
        rotations.row(i) = rot;
    }
    return rotations;
}

arma::mat opt_compute_problem(int frame, params p, const OptData& data,
                              const arma::mat& rotations) {
    const auto& flow = data.flows.data.at(frame);
    arma::mat ap = flow.rows(0, 2);
    arma::mat bp = flow.rows(3, 5);
    arma::mat problem(flow.n_cols, 3);
    p.align() = safe_normalize(p.align());
    for (int i = 0; i < flow.n_cols; ++i) {
        arma::vec4 rot = quat_prod(quat_conj(p.align()), quat_prod(rotations.row(i), p.align()));
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

int main() {
    std::cout << std::fixed << std::setprecision(15);

    OptData opt_data;
    optdata_fill_gyro(opt_data, "GX011338.MP4", "yZX");
    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 91);

    arma::mat P;
    opt_compute_problem(90, -42, opt_data, P);
    arma::mat t = opt_guess_translational_motion(P);

    FrameState fs(90, &opt_data);
    fs.motion_vec = t;
    fs.var_k = 1e3;
    fs.gyro_delay = {-42};

    // arma::mat cost, jac_delay, jac_motion;

    // int n = 10000;
    // fs.motion_vec[0] -= 1e-3 * n / 2;
    // for (int i = 0; i < 10000; i++) {
    //     fs.Cost(fs.gyro_delay, fs.motion_vec, cost, jac_delay, jac_motion);

    //     std::cout << fs.motion_vec[0] << "," << cost[0] << std::endl;
    //     // std::cout << fs.motion_vec[0] << "," << jac_motion[0] << std::endl;

    //     fs.motion_vec[0] += 1e-3;
    // }

    Gsl::MultiminFunction func(3);
    func.SetF([&fs](arma::vec x) {
        arma::mat cost;
        fs.CostOnly(fs.gyro_delay, x, cost);
        return cost[0];
    });
    func.SetFdF([&fs](arma::vec x) {
        arma::mat cost, del_jac, mot_jac;
        fs.Cost(fs.gyro_delay, x, cost, del_jac, mot_jac);
        return std::make_pair(cost[0], arma::vec{mot_jac.t()});
    });
    gsl_vector* motion_vec = gsl_vector_alloc(3);

    gsl_multimin_fdfminimizer* minimizer =
        gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_vector_bfgs2, 3);

    arma::wrap(motion_vec) = fs.motion_vec;
    gsl_multimin_fdfminimizer_set(minimizer, &func.gsl_func, motion_vec, 1e-2, 1e-4);

    for (int j = 0; j < 50; ++j) {
        auto r = gsl_multimin_fdfminimizer_iterate(minimizer);
        std::cout << arma::wrap(minimizer->gradient) << std::endl;
        // if (r != GSL_CONTINUE) {
        //     std::cerr << j << " " << r << std::endl;
        //     std::cerr << "stop " << j << std::endl;
        //     break;
        // }
    }
}
