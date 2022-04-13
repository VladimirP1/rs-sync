// The plots of SVD-based cost vs optimization-based still do not match exactly ...

#include <iomanip>
#include <iostream>
#include <vector>

#include "cv.hpp"
#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "loss.hpp"
#include "io.hpp"

#include "inline_utils.hpp"

static double calc(arma::mat P, arma::mat M, double k) {
    arma::mat r = (P * M) * (k / arma::norm(M));
    arma::mat rho = arma::log1p(r % r);
    return sqrt(arma::accu(arma::sqrt(rho)));
}


arma::vec3 opt_guess_translational_motion2(const arma::mat& problem) {
    arma::mat nproblem = problem;
    nproblem.each_row([](arma::mat& m) { m = safe_normalize(m); });

    arma::vec3 best_sol;
    double least_med = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 20; ++i) {
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

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(16);

    OptData opt_data;
    // YXZ yZX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "yZX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    track_frames(opt_data.flows, lens, "GX011338.MP4", 90, 90 + 30);

    // const double k = 1e3;
    for (double delay = -2000; delay < 2000; delay += 2) {
        double cost = 0;
        double cost2 = 0;
        double cost3 = 0;
        const bool fast = true;
        for (auto& [frame, _] : opt_data.flows.data) {
            arma::mat P, M;
            opt_compute_problem(frame, delay, opt_data, P);
            M = opt_guess_translational_motion2(P);
            double k = 1 / arma::norm(P * M) * 1e2;
            // std::cerr << k << std::endl;

            cost3 += calc(P, M, k);

            if (!fast) {
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
        }
        std::cout << delay << "," << cost3 << "," << cost << "," << cost2 << std::endl;
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
