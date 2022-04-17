#include "loss.hpp"

#include "ndspline.hpp"
#include "quat.hpp"
#include "inline_utils.hpp"
#include "optdata.hpp"

void opt_compute_problem(int frame, double gyro_delay, const OptData& data, arma::mat& problem) {
    const auto& flow = data.flows.data.at(frame);
    gyro_delay /= 1000.;

    arma::mat ap = flow.rows(0, 2);
    arma::mat bp = flow.rows(3, 5);
    double baset = ((frame / data.flows.fps) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat at = (flow.row(6) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat bt = (flow.row(7) - data.quats_start + gyro_delay) * data.sample_rate;

    problem.resize(at.n_cols, 3);
    arma::vec4 base_conj = quat_conj(data.quats.eval(baset));
    for (int i = 0; i < at.n_cols; ++i) {
        arma::vec4 a = data.quats.eval(at[i]);
        arma::vec4 b = data.quats.eval(bt[i]);
        double inv_base_a_norm = (1. / arma::norm(base_conj)) * (1. / arma::norm(a));
        double inv_base_b_norm = (1. / arma::norm(base_conj)) * (1. / arma::norm(b));
        arma::vec4 rot_a = quat_prod(base_conj, a) * inv_base_a_norm;
        arma::vec4 rot_b = quat_prod(base_conj, b) * inv_base_b_norm;
        arma::vec3 ar = quat_rotate_point(quat_conj(rot_a), ap.col(i));
        arma::vec3 br = quat_rotate_point(quat_conj(rot_b), bp.col(i));
        problem.row(i) = arma::trans(arma::cross(ar, br));
    }
}

arma::vec3 opt_guess_translational_motion(const arma::mat& problem, int max_iters) {
    arma::mat nproblem = problem;
    nproblem.each_row([](arma::mat& m) { m = safe_normalize(m); });

    arma::vec3 best_sol;
    double least_med = std::numeric_limits<double>::infinity();
    for (int i = 0; i < max_iters; ++i) {
        int vs[2];
        vs[0] = vs[1] = mtrand(0, problem.n_rows - 1);
        while (vs[1] == vs[0]) vs[1] = mtrand(0, problem.n_rows - 1);

        arma::mat v =
            arma::trans(safe_normalize(arma::cross(problem.row(vs[0]), problem.row(vs[1]))));

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

std::pair<double, double> pre_sync(OptData& opt_data, int frame_begin, int frame_end, double rough_delay,
                double search_radius, double step) {
    std::vector<std::pair<double, double>> results;
    for (double delay = rough_delay - search_radius; delay < rough_delay + search_radius;
         delay += step) {
        double cost = 0;
        for (auto& [frame, _] : opt_data.flows.data) {
            if (frame < frame_begin || frame >= frame_end) continue;
            arma::mat P, M;
            opt_compute_problem(frame, delay, opt_data, P);
            M = opt_guess_translational_motion(P, 20);
            double k = 1 / arma::norm(P * M) * 1e2;
            arma::mat r = (P * M) * (k / arma::norm(M));
            arma::mat rho = arma::log1p(r % r);
            cost += sqrt(arma::accu(arma::sqrt(rho)));
        }
        results.emplace_back(cost, delay);
    }
    return *std::min_element(results.begin(), results.end());
}