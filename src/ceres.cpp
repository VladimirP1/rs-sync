#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "cv.hpp"

#include <telemetry-parser.h>

#include <spline.hpp>

#include <ceres/ceres.h>

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

void opt_compute_problem(int frame, double gyro_delay, const OptData& data, arma::mat& problem,
                         arma::mat& dproblem) {
    gyro_delay /= 1000;
    const auto& flow = data.flows.at(frame);

    arma::mat ap = flow.rows(0, 2);
    arma::mat bp = flow.rows(3, 5);
    arma::mat at = (flow.row(6) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat bt = (flow.row(7) - data.quats_start + gyro_delay) * data.sample_rate;

    problem.resize(at.n_cols, 3);
    dproblem.resize(at.n_cols, 3);
    for (int i = 0; i < at.n_cols; ++i) {
        // Gyro integration with detivative wrt time offset
        arma::vec4 a_conj = quat_conj(data.quats.eval(at[i]));
        arma::vec4 da_conj = quat_conj(data.quats.deriv(at[i]));
        arma::vec4 b = data.quats.eval(bt[i]);
        arma::vec4 db = data.quats.deriv(bt[i]);
        double inv_ab_norm = (1. / arma::norm(a_conj)) * (1. / arma::norm(b));
        arma::vec4 rot = quat_prod(a_conj, b) * inv_ab_norm;
        arma::vec4 drot =
            (quat_prod(da_conj, b) + quat_prod(a_conj, db)) * inv_ab_norm;  // TODO: This is wrong
        drot -= drot * arma::dot(rot, drot);

        arma::vec3 br = quat_rotate_point(quat_conj(rot), bp.col(i));
        arma::vec3 t = arma::cross(br, drot.rows(1, 3));
        problem.row(i) = arma::trans(arma::cross(ap.col(i), br));
        dproblem.row(i) = arma::trans(arma::cross(ap.col(i), t)) * data.sample_rate;
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
struct ComputeProblemFunctor {
    ComputeProblemFunctor(int frame, OptData* optdata) : frame_{frame}, optdata_{optdata} {}
    bool operator()(double const* gyro_delay, double* value) const {
        arma::mat problem(value, optdata_->flows[frame_].n_cols, 3, false, true), dproblem;
        opt_compute_problem(frame_, *gyro_delay, *optdata_, problem, dproblem);
        return true;
    }

   private:
    int frame_;
    OptData* optdata_;
};

struct FrameState : public ceres::SizedCostFunction<1, 1, 3> {
    FrameState(int frame, OptData* optdata)
        : frame_{frame}, optdata_{optdata}, problem_functor_(frame, optdata) {}

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

    static double calc(arma::mat P, arma::mat M, double k) {
        auto [v1, j1] = std::make_tuple(P * M, P);
        auto [v2, j2] = sqr_jac(v1);

        auto [v3, j3] = sqr_jac(M);
        auto [v4, j4] = sum_jac(v3);
        auto [v5, j5, _] = div_jac(v4, k * k);

        auto [v6, j6a, j6b] = div_jac(v2, v5[0]);
        auto [v7, j7] = log1p_jac(v6);
        auto [v8, j8] = sum_jac(v7);

        return v8[0];
    }

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        const double* gyro_delay = parameters[0];
        const double* motion_raw = parameters[1];
        arma::mat M(motion_raw, 3, 1);

        arma::mat P, P2, tmp;

        opt_compute_problem(frame_, *gyro_delay, *optdata_, P, tmp);
        opt_compute_problem(frame_, *gyro_delay + kStep, *optdata_, P2, tmp);

        double r1 = calc(P, M, k);
        double r2 = calc(P2, M, k);

        auto [v1, j1] = std::make_tuple(P * M, P);
        auto [v2, j2] = sqr_jac(v1);

        auto [v3, j3] = sqr_jac(M);
        auto [v4, j4] = sum_jac(v3);
        auto [v5, j5, _] = div_jac(v4, k * k);

        auto [v6, j6a, j6b] = div_jac(v2, v5[0]);
        auto [v7, j7] = log1p_jac(v6);
        auto [v8, j8] = sum_jac(v7);

        residuals[0] = v8[0];

        if (jacobians) {
            if (jacobians[0]) {
                jacobians[0][0] = (r2 - r1) / kStep;
            }
            if (jacobians[1]) {
                arma::mat drho_dM(jacobians[1], 1, 3, false, true);
                drho_dM = j8 * j7 * (j6a * j2 * j1 + j6b * j5 * j4 * j3);
            }
        }

        return true;
    }

    void GuessMotion(double gyro_delay) {
        arma::mat problem(optdata_->flows[frame_].n_cols, 3);
        // double* g = &gyro_delay;
        problem_functor_.operator()(&gyro_delay, problem.memptr());
        motion_vec = opt_guess_translational_motion(problem);
    }

    arma::vec3 motion_vec;

   private:
    static constexpr double k = 1e3;
    static constexpr double kStep = 1e-5;

    int frame_;
    OptData* optdata_;

    ComputeProblemFunctor problem_functor_;
};

arma::mat compute_rho(arma::mat problem, arma::vec3 sol) {
    static constexpr double k = 1e2;

    arma::mat residuals = problem * sol;
    arma::mat rho = log(1 + (residuals * k) % (residuals * k)) / (k * k) / 2;
    return rho;
}

void opt_run(OptData& data) {
    double gyro_delay = -47;
    ceres::Problem p;

    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    // options.use_inner_iterations = true;
    options.use_nonmonotonic_steps = true;
    // options.inner_iteration_tolerance = 1e-4;
    // options.max_trust_region_radius = 1e-3;
    options.function_tolerance = 1e-9;
    // options.initial_trust_region_radius = 1e-4;
    options.line_search_sufficient_function_decrease = .2;
    // options.num_threads = 8;
    options.logging_type = ceres::SILENT;
    options.minimizer_type = ceres::LINE_SEARCH;
    // options.check_gradients = true;

    std::vector<FrameState*> costs;
    for (auto& [frame, _] : data.flows) {
        auto raw_cost_func = new FrameState(frame, &data);
        costs.push_back(raw_cost_func);
        // ceres::CostFunction* frame_cost =
        //     new ceres::NumericDiffCostFunction<FrameCostFunction, ceres::FORWARD, 1, 1, 3>(
        //         raw_cost_func);
        ceres::CostFunction* frame_cost = raw_cost_func;
        raw_cost_func->GuessMotion(gyro_delay);
        p.AddResidualBlock(frame_cost, nullptr, &gyro_delay, raw_cost_func->motion_vec.memptr());
    }

    for (int i = 0; i < 20; i++) {
        ceres::Solver::Summary summary;
        ceres::Solve(options, &p, &summary);
        // std::cout << summary.FullReport() << "\n";
        std::cout << gyro_delay << " " << summary.final_cost << std::endl;

        for (auto raw_cost : costs) {
            raw_cost->GuessMotion(gyro_delay);
        }
    }

    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &p, &summary);
    // std::cout << summary.FullReport() << "\n";
}

std::vector<arma::vec3> motion;
double cost(const OptData& data, double gyro_delay, double& der_sum) {
    double cost_sum = 0;
    der_sum = 0;

    static constexpr double k = 1e3;

    bool first_run = motion.empty();
    motion.resize(data.flows.size());

    for (int iter = 0; iter < 1; ++iter) {
        int frame = 0;
        for (auto& [frame_id, _] : data.flows) {
            arma::mat problem, dproblem;
            opt_compute_problem(frame_id, gyro_delay, data, problem, dproblem);

            motion[frame] = opt_guess_translational_motion(problem);

            // arma::mat c0 = problem * motion[frame];
            arma::mat c0 = compute_rho(problem, motion[frame]);

            opt_compute_problem(frame_id, gyro_delay - 1e-7, data, problem, dproblem);
            // arma::mat c1 = problem * motion[frame];
            arma::mat c1 = compute_rho(problem, motion[frame]);

            opt_compute_problem(frame_id, gyro_delay + 1e-7, data, problem, dproblem);
            // arma::mat c2 = problem * motion[frame];
            arma::mat c2 = compute_rho(problem, motion[frame]);

            // cost_sum += (arma::accu(c0 % c0));
            // der_sum += ((arma::accu(c2 % c2)) - (arma::accu(c1 % c1))) / 1e-7 / 2;
            cost_sum += (arma::accu(c0));
            der_sum += ((arma::accu(c2)) - (arma::accu(c1))) / 1e-7 / 2;
            ++frame;
        }
    }
    return cost_sum;
}

int main() {
    std::cout << std::fixed << std::setprecision(16);

    OptData opt_data;
    // YXZ zYX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "zYX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    track_frames(opt_data.flows, lens, "GX011338.MP4", 400, 450);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 400, 450);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1300, 1330);

    opt_run(opt_data);

    // for (double ofs = -.1; ofs < .1; ofs += 1e-3) {
    // for (double ofs = -.065; ofs < -.035; ofs += 1e-4) {
    //     // double der;
    //     // std::cout << ofs << "," << cost(opt_data, ofs, der) << "," << der << std::endl;
    //     arma::mat p,dp;
    //     opt_compute_problem(300, ofs + 1e-7, opt_data, p, dp);
    //     double r = p[2];
    //     opt_compute_problem(300, ofs, opt_data, p, dp);
    //     double l = p[2];
    //     std::cout << ofs << "," << dp[2] << "," << (r - l) / 1e-7 << std::endl;
    // }
    // arma::vec4 a_conj = quat_conj(nsp.eval(t_left));
    // arma::vec4 b = nsp.eval(t_right);
    // arma::vec4 da_conj = quat_conj(nsp.deriv(t_left));
    // arma::vec4 db = nsp.deriv(t_right);
    // i2 = quat_prod(a_conj, b) / arma::norm(a_conj) / arma::norm(b);
    // di2 = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) / arma::norm(a_conj) /
    //       arma::norm(b);
    // di2 -= i2 * arma::dot(i2, di2);
}
