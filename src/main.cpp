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

arma::vec4 gyro_integrate(const ndspline& nsp, double t1, double t2) {
    arma::vec4 p = arma::normalise(nsp.eval(t1));
    arma::vec4 q = arma::normalise(nsp.eval(t2));
    return quat_prod(quat_conj(p), q);
}

int main() {
    std::cout << std::fixed << std::setprecision(16);

    tp_gyrodata data = tp_load_gyro("GX011338.MP4", "XYZ");
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    int sr = gyro_interpolate(timestamps, gyro);

    for (int i = 0; i < data.samples; ++i) {
        gyro.col(i) = arma::randn(3, 1) * 1;
    }

    arma::mat quats(4, data.samples);
    quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < quats.n_cols; ++i) {
        quats.col(i) = arma::normalise(quat_prod(quats.col(i - 1), quat_from_aa(gyro.col(i) / sr)));
    }

    auto nsp = ndspline::make(quats);

    const double eps = 1e-7;
    const double win = 1. / 30;
    for (double t = 95; t < 150; t += .001) {
        const double t_left = t;
        const double t_right = t + win;
        arma::vec4 di1, di2, i1, i2;
        {
            arma::vec4 l = gyro_integrate(nsp, t_left, t_right);
            arma::vec4 r = gyro_integrate(nsp, t_left + eps, t_right + eps);
            di1 = quat_prod(quat_conj(l), r) / eps;
            i1 = l;
        }
        {
            arma::vec4 a_conj = quat_conj(nsp.eval(t_left));
            arma::vec4 b = nsp.eval(t_right);
            arma::vec4 da_conj = quat_conj(nsp.deriv(t_left));
            arma::vec4 db = nsp.deriv(t_right);
            i2 = quat_prod(a_conj, b) / arma::norm(a_conj) / arma::norm(b);
            di2 = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) / arma::norm(a_conj) /
                  arma::norm(b);
            di2 -= i2 * arma::dot(i2, di2);
            // std::cout << arma::dot(di2, i2) << std::endl;

            // std::cout << acos(arma::dot(b / arma::norm(b), db / arma::norm(b)) / arma::norm(b) /
            //                   arma::norm(db)) *
            //                  180 / M_PI
            //           << std::endl;
        }

        // std::cout << t << "," << i1[1] << "," << i2[1] << std::endl;
        std::cout << t << "," << di1[1] << "," << di2[1] << std::endl;
        // std::cout << t << "," << di2[0] << "," << di2[1] << "," << di2[1]/di2[0] << std::endl;
    }
}
