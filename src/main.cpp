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
        arma::mat dproblem(at.n_cols, 3);
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
            arma::vec3 t = arma::cross(br, drot.rows(1,3));
            problem.row(i) = arma::trans(arma::cross(ap.col(i), br));
            dproblem.row(i) = arma::trans(arma::cross(ap.col(i), t));
            brp.col(i) = br;
            rots.col(i) = rot;
        }

        for (int i = 0; i < problem.n_rows; ++i) {
            std::cout << problem(i, 0) << " " << problem(i, 1) << " " << problem(i, 2) << " " << dproblem(i, 0) << " " << dproblem(i, 1) << " " << dproblem(i, 2) << std::endl;
        }
        exit(0);
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(16);

    OptData opt_data;
    optdata_fill_gyro(opt_data, "GX011338.MP4", "XYZ");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    track_frames(opt_data.flows, lens, "GX011338.MP4", 304, 305);

    opt_run(opt_data);

    // arma::vec4 a_conj = quat_conj(nsp.eval(t_left));
    // arma::vec4 b = nsp.eval(t_right);
    // arma::vec4 da_conj = quat_conj(nsp.deriv(t_left));
    // arma::vec4 db = nsp.deriv(t_right);
    // i2 = quat_prod(a_conj, b) / arma::norm(a_conj) / arma::norm(b);
    // di2 = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) / arma::norm(a_conj) /
    //       arma::norm(b);
    // di2 -= i2 * arma::dot(i2, di2);
}
