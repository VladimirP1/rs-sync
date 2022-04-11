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
#include "io.hpp"
#include "gsl.hpp"

#include <telemetry-parser.h>


inline void nms(double* data, int nsamples, int ndim, int radius) {
    std::vector<double> newdata(ndim * nsamples);
    for (int dim = 0; dim < ndim; ++dim) {
        for (int i = 0; i < nsamples; ++i) {
            double cur = data[ndim * i + dim];
            newdata[ndim * i + dim] = cur;
            for (int j = std::max(0, i - radius); j < std::min(nsamples, i + radius); ++j) {
                if (data[ndim * j + dim] > cur) {
                    newdata[ndim * i + dim] = 0;
                }
            }
        }
    }
    std::copy(newdata.begin(), newdata.end(), data);
}

int main() {
    std::cout << std::fixed << std::setprecision(14);

    tp_gyrodata data = tp_load_gyro("GX011338.MP4", "yZX");
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    tp_free(data);

    int sample_rate = gyro_interpolate(timestamps, gyro);
    int lpf1 = 20;
    int lpf2 = 1000;
    // int radius = 5000;

    gyro_lowpass(gyro, lpf1);

    arma::vec3 sobel{-1, 0, 1};
    gyro.row(0) = arma::conv(gyro.row(0), sobel, "same");
    gyro.row(1) = arma::conv(gyro.row(1), sobel, "same");
    gyro.row(2) = arma::conv(gyro.row(2), sobel, "same");
    gyro = gyro % gyro;

    gyro_lowpass(gyro, lpf2);

    for (int i = 0; i < gyro.n_cols; ++i) {
        std::cout << i / 200. << "," << std::max(gyro(0,i),0.0) + std::max(gyro(1,i),0.0) + std::max(gyro(2,i),0.0) << std::endl;
    }
}