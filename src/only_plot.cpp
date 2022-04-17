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

static constexpr const char* variants[] = {
    "YxZ", "Xyz", "XZy", "Zxy", "zyX", "yxZ", "ZXY", "zYx", "ZYX", "yXz", "YZX", "XyZ",
    "Yzx", "zXy", "YXz", "xyz", "yZx", "XYZ", "zxy", "xYz", "XYz", "zxY", "zXY", "xZy",
    "zyx", "xyZ", "Yxz", "xzy", "yZX", "yzX", "ZYx", "xYZ", "zYX", "ZxY", "yzx", "xZY",
    "Xzy", "XzY", "YzX", "Zyx", "XZY", "yxz", "xzY", "ZyX", "YXZ", "yXZ", "YZx", "ZXy"};

void find_orient() {
    OptData opt_data;
    Lens lens = lens_load("lens.txt", "xlite4k43");
    track_frames(opt_data.flows, lens, "171836AA.MP4", 30, 30 + 60);

    for (int i = 0; i < sizeof(variants); ++i) {
        optdata_fill_gyro(opt_data, "171836AA.CSV", variants[i]);
        double cost = 0;
        for (auto& [frame, _] : opt_data.flows.data) {
            arma::mat P, M;
            opt_compute_problem(frame, -34, opt_data, P);
            M = opt_guess_translational_motion(P, 20);
            double k = 1 / arma::norm(P * M) * 1e2;
            cost += calc(P, M, k);
        }
        std::cout << variants[i] << " " << cost << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(16);

    // find_orient();
    // return 0;

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
            M = opt_guess_translational_motion(P, 20);
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
