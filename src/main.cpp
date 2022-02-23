#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

#include "quat.hpp"
#include "signal.hpp"
#include "cv.hpp"

#include <telemetry-parser.h>

struct OptData {
    double quats_ts{};
    int sample_rate{};
    arma::mat quats{};

    arma::mat features{};
};

void optdata_fill_gyro(OptData& optdata, const char* filename, const char* orient) {
    tp_gyrodata data = tp_load_gyro(filename, orient);
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    optdata.sample_rate = gyro_interpolate(timestamps, gyro);

    optdata.quats.resize(4, data.samples);
    optdata.quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < optdata.quats.n_cols; ++i) {
        optdata.quats.col(i) =
            arma::normalise(quat_prod(optdata.quats.col(i - 1), quat_from_aa(gyro.col(i) / optdata.sample_rate)));
    }
    optdata.quats_ts = timestamps.front();
}

void optdata_fill_optflow(OptData& optdata, const FramesFlow& flow, int start_frame, int end_frame) {

}


int main() {
    OptData opt_data;
    optdata_fill_gyro(opt_data, "GX011338.MP4", "XYZ");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    FramesFlow flow;
    track_frames(flow, lens, "GX011338.MP4", 180, 180 + 60);

    optdata_fill_optflow(opt_data, flow, 180, 180 + 60);
}

int main2() {
    std::cout << std::fixed << std::setprecision(16);

    tp_gyrodata data = tp_load_gyro("GX011338.MP4", "XYZ");
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    int sr = gyro_interpolate(timestamps, gyro);

    arma::mat quats(4, data.samples);
    quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < quats.n_cols; ++i) {
        quats.col(i) = arma::normalise(quat_prod(quats.col(i - 1), quat_from_aa(gyro.col(i) / sr)));
    }
    // std::cout << sr << std::endl;

    // std::cout << quats << std::endl;

    int cnt = 0;
    auto start = std::chrono::steady_clock::now();
    for (double t = 0; t < 1; t += .0000001) {
        int idx = static_cast<int>(t);
        quat_quad(quats.col(idx + 0), quats.col(idx + 1), quats.col(idx + 2), quats.col(idx + 3),
                  t - idx);
        // std::cout << arma::norm(quat_to_aa(quat_squad(quats[0], quats[1], quats[2], quats[3],
        // t)))
        //   << std::endl;
        // std::cout << arma::norm(quat_to_aa(quat_slerp(quats[1], quats[2], t)))
        //   << std::endl;
        // std::cout << arma::norm(quat_to_aa(quats[1]) * (1-t) +
        // t*quat_to_aa(quats[2])) << std::endl;
        ++cnt;
    }
    // auto end = std::chrono::steady_clock::now();
    // std::cout << cnt/std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() *
    // 1000 << std::endl;

    std::cout << timestamps.back() << std::endl;
    return 0;

    for (double t = 1; t < 5700; t += .01) {
        int idx = static_cast<int>(t);
        std::cout << t << ","
                  << quat_to_aa(quat_squad(quats.col(idx - 1), quats.col(idx + 0),
                                           quats.col(idx + 1), quats.col(idx + 2), t - idx))[0]
                  << ","
                  << quat_to_aa(quat_quad(quats.col(idx - 1), quats.col(idx + 0),
                                          quats.col(idx + 1), quats.col(idx + 2), t - idx))[0]
                  << "," << quat_to_aa(quats.col(idx))[0] << "," << idx << std::endl;
    }
}