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
        arma::vec4 drot = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) * inv_ab_norm;
        drot -= drot * arma::dot(rot, drot);

        arma::vec3 br = quat_rotate_point(quat_conj(rot), bp.col(i));
        arma::vec3 t = arma::cross(br, drot.rows(1, 3));
        problem.row(i) = arma::trans(arma::cross(ap.col(i), br));
        dproblem.row(i) = arma::trans(arma::cross(ap.col(i), t));
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

arma::mat compute_rho(arma::mat problem, arma::vec3 sol) {
    static constexpr double k = 1e3;

    arma::mat residuals = problem * sol;
    arma::mat rho = log(1 + (residuals * k) % (residuals * k)) / (k * k) / 2;
    return rho;
}

void opt_run(const OptData& data) {
    double gyro_delay = -.054;

    static constexpr double k = 1e3;

    std::vector<arma::vec3> motion(data.flows.size());
    for (int iter = 0; iter < 100; ++iter) {
        int frame = 0;
        for (auto& [frame_id, _] : data.flows) {
            arma::mat problem, dproblem;
            opt_compute_problem(frame_id, gyro_delay, data, problem, dproblem);

            motion[frame] = motion[frame] * .9 + .1 * opt_guess_translational_motion(problem);

            arma::mat residuals = problem * motion[frame];
            arma::mat rho = log(1 + (residuals * k) % (residuals * k)) / (k * k) / 2;

            ++frame;
        }
    }
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
    // YXZ yZX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "yZX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 300, 330);
    track_frames(opt_data.flows, lens, "GX011338.MP4", 400, 415);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1300, 1330);

    // opt_run(opt_data);

    // for (double ofs = -.1; ofs < .1; ofs += 1e-3) {
    for (double ofs = -.065; ofs < -.035; ofs += 1e-4) {
        double der;
        std::cout << ofs << "," << cost(opt_data, ofs, der) << "," << der << std::endl;
    }
    // arma::vec4 a_conj = quat_conj(nsp.eval(t_left));
    // arma::vec4 b = nsp.eval(t_right);
    // arma::vec4 da_conj = quat_conj(nsp.deriv(t_left));
    // arma::vec4 db = nsp.deriv(t_right);
    // i2 = quat_prod(a_conj, b) / arma::norm(a_conj) / arma::norm(b);
    // di2 = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) / arma::norm(a_conj) /
    //       arma::norm(b);
    // di2 -= i2 * arma::dot(i2, di2);
}
