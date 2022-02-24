#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

#include "quat.hpp"
#include "signal.hpp"
#include "cv.hpp"

#include <telemetry-parser.h>

struct OptData {
    double quats_start{};
    int sample_rate{};
    arma::mat quats{};

    FramesFlow flows{};
};

void optdata_fill_gyro(OptData& optdata, const char* filename, const char* orient) {
    tp_gyrodata data = tp_load_gyro(filename, orient);
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    tp_free(data);
    optdata.sample_rate = gyro_interpolate(timestamps, gyro);

    optdata.quats.resize(4, data.samples);
    optdata.quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < optdata.quats.n_cols; ++i) {
        optdata.quats.col(i) = arma::normalise(
            quat_prod(optdata.quats.col(i - 1), quat_from_aa(gyro.col(i) / optdata.sample_rate)));
    }
    optdata.quats_start = timestamps.front();
}

arma::vec4 gyro_integrate(const arma::mat& quats, double from, double to) {
    // std::tie(from, to) = std::make_pair(std::max(from, 1.), std::min(to, quats.n_cols - 3.));
    const auto integr = [&](int x, int y) {
        int nx{y-x}, ny{y + x};
        return quat_prod(quat_conj(quats.col(from + nx)), quats.col(to + ny));
    };
    arma::vec4 curv0, curv1, curv2, curv3;
    {
        arma::vec4 p0 = integr(-1, -1);
        arma::vec4 p1 = integr(+0, -1);
        arma::vec4 p2 = integr(+1, -1);
        arma::vec4 p3 = integr(+2, -1);
        curv0 = quat_quad(p0, p1, p2, p3, from - static_cast<int>(from));
    }
    {
        arma::vec4 p0 = integr(-1, +0);
        arma::vec4 p1 = integr(+0, +0);
        arma::vec4 p2 = integr(+1, +0);
        arma::vec4 p3 = integr(+2, +0);
        curv1 = quat_quad(p0, p1, p2, p3, from - static_cast<int>(from));
    }
    {
        arma::vec4 p0 = integr(-1, +1);
        arma::vec4 p1 = integr(+0, +1);
        arma::vec4 p2 = integr(+1, +1);
        arma::vec4 p3 = integr(+2, +1);
        curv2 = quat_quad(p0, p1, p2, p3, from - static_cast<int>(from));
    }
    {
        arma::vec4 p0 = integr(-1, +2);
        arma::vec4 p1 = integr(+0, +2);
        arma::vec4 p2 = integr(+1, +2);
        arma::vec4 p3 = integr(+2, +2);
        curv3 = quat_quad(p0, p1, p2, p3, from - static_cast<int>(from));
    }
    return quat_quad(curv0, curv1, curv2, curv3, to - static_cast<int>(to));
}

arma::vec4 gyro_integrate_grad(const arma::mat& quats, double from, double to) {
    const double step = 1e-10;
    // under small angle assumption, sinx=x
    arma::vec4 mid = gyro_integrate(quats, from, to);
    arma::vec4 left = gyro_integrate(quats, from - step, to - step);
    arma::vec4 right = gyro_integrate(quats, from + step, to + step);
    arma::vec4 grad =
        (quat_prod(quat_conj(left), mid) + quat_prod(quat_conj(mid), right)) / (2 * step);
    return grad;
}

void opt_run(const OptData& data) {
    double gyro_delay = -.045;
    for (const auto& [_, flow] : data.flows) {
        arma::mat ap = flow.rows(0, 2);
        arma::mat bp = flow.rows(3, 5);
        arma::mat at = (flow.row(6) - data.quats_start + gyro_delay) * data.sample_rate;
        arma::mat bt = (flow.row(7) - data.quats_start + gyro_delay) * data.sample_rate;

        arma::mat rots(4, at.n_cols);
        arma::mat brp(3, at.n_cols);
        arma::mat problem(at.n_cols, 3);
        for (int i = 0; i < at.n_cols; ++i) {
            arma::vec4 rot = gyro_integrate(data.quats, at[i], bt[i]);
            arma::vec3 br = quat_rotate_point(quat_conj(rot), bp.col(i));
            problem.row(i) = arma::trans(arma::cross(ap.col(i), br));
            brp.col(i) = br;
            rots.col(i) = rot;
        }

        for (int i = 0; i < problem.n_rows; ++i) {
            std::cout << problem(i, 0) << " " << problem(i, 1) << " " << problem(i, 2) << std::endl;
        }
        // exit(0);
    }
}

int main() {
    OptData opt_data;
    optdata_fill_gyro(opt_data, "GX011338.MP4", "XYZ");

    // Lens lens = lens_load("lens.txt", "hero6_27k_43");
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 304, 305);

    // opt_run(opt_data);

    for (double t = 3; t < 20; t += .001) {
        arma::vec4 g = gyro_integrate(opt_data.quats, t, t + 1./30);
        std::cout << t << "," << g[2] << std::endl;
    }
}
