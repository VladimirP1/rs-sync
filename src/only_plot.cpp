// The plots of SVD-based cost vs optimization-based still do not match exactly ...

#include <iomanip>
#include <iostream>
#include <vector>

#include "cv.hpp"
#include "inline_utils.hpp"
#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"

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
    for (int i = 0; i < problem.n_rows; ++i) {
        nproblem.row(i) = arma::normalise(nproblem.row(i));
    }

    arma::vec3 best_sol;
    double least_med = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 200; ++i) {
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
static double calc(arma::mat P, arma::mat M, double k) {
    arma::mat r = (P * M) * (k / arma::norm(M));
    arma::mat rho = arma::log1p(r % r);
    return sqrt(arma::accu(arma::sqrt(rho)));
}

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(16);

    OptData opt_data;
    // YXZ yZX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "yZX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 90 + 30);

    // const double k = 1e3;
    for (double delay = -60e-3; delay < -30e-3; delay += 1e-4) {
        double cost = 0;
        double cost2 = 0;
        for (auto& [frame, _] : opt_data.flows) {
            arma::mat P, M;
            opt_compute_problem(frame, delay, opt_data, P);
            M = opt_guess_translational_motion(P);
            double k = 1 / arma::norm(P * M) * 1e1;
            std::cerr << k << std::endl;

            arma::mat residuals = (P * M);
            arma::mat weights = arma::sqrt(1 / (1 + (residuals % residuals) * k * k));
            for (int i = 0; i < 20; ++i) {
                arma::vec S;
                arma::mat U, V;
                arma::svd(U, S, V, P.each_col() % weights, "std");

                residuals = (P * V.col(V.n_cols - 1));
                weights = arma::sqrt(1 / (1 + (residuals % residuals) * k * k));
            }

            arma::vec S;
            arma::mat U, V;
            arma::svd(U, S, V, P.each_col() % weights, "std");

            arma::mat sol = V.col(V.n_cols - 1);

            cost += fabs(S[S.n_rows - 1]);
            cost2 += fabs(calc(P, sol, k));
        }
        std::cout << delay << "," << cost << "," << cost2 << std::endl;
    }

    // arma::mat out;
    // opt_compute_problem(90, -44.1e-3, opt_data, out);

    // for (int row = 0;row < out.n_rows;++row) {
    //     for (int col = 0; col < out.n_cols; ++col) {
    //         std::cout << out(row,col) << " ";
    //     }
    //     std::cout << std::endl;
    // }
}
