#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

#include "quat.hpp"
#include "signal.hpp"
#include "cv.hpp"

#include <telemetry-parser.h>

#include <spline.hpp>

struct ndspline {
    std::vector<tk::spline> splines;
};

ndspline ndspline_make(const arma::mat& m) {
    ndspline ret;
    std::vector<double> X(m.n_cols);
    std::generate_n(X.begin(), m.n_cols, [i = 0]() mutable { return i++; });
    for (int row = 0; row < m.n_rows; ++row) {
        std::vector<double> Y(m.n_cols);
        std::copy(m.begin_row(row), m.end_row(row), Y.begin());
        ret.splines.push_back(tk::spline(X, Y));
    }
    return ret;
}

arma::mat ndspline_eval(const ndspline& nsp, double t) {
    arma::mat ret(nsp.splines.size(), 1);
    for (int i = 0; i < ret.size(); ++i) {
        ret[i] = nsp.splines[i](t);
    }
    return ret;
}

arma::mat ndspline_der(const ndspline& nsp, double t) {
    arma::mat ret(nsp.splines.size(), 1);
    for (int i = 0; i < ret.size(); ++i) {
        ret[i] = nsp.splines[i].deriv(1, t);
    }
    return ret;
}

arma::vec4 gyro_integrate(const ndspline& nsp, double t1, double t2) {
    // arma::vec4 p = arma::normalise(ndspline_eval(nsp, t1));
    // arma::vec4 q = arma::normalise(ndspline_eval(nsp, t2));
    arma::vec4 p = arma::normalise(ndspline_eval(nsp, t1));
    arma::vec4 q = arma::normalise(ndspline_eval(nsp, t2));
    return quat_prod(quat_conj(p), q);
    // return q - p;
}

int main() {
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

    auto nsp = ndspline_make(quats);

    for (double t = 95; t < 100; t += .001) {
        arma::vec4 di, di2;
        {
            arma::vec4 i_l = gyro_integrate(nsp, t, t + 1./30);
            arma::vec4 i_r = gyro_integrate(nsp, t + 1e-9, t + 1./30 + 1e-9);
            di = quat_prod(quat_conj(i_l), i_r) / 1e-9;
        }
        {
            arma::vec4 q_l  = ndspline_eval(nsp, t);
            arma::vec4 q_r  = ndspline_eval(nsp, t + 1./30);
            di2 = quat_prod(quat_conj(ndspline_der(nsp, t)), q_r) + quat_prod(quat_conj(q_l), ndspline_der(nsp, t + 1./30));
        }
        std::cout << t << "," << di[1] << "," << di2[1] << std::endl;
    }
}
