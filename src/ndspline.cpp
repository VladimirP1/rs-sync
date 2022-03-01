#include "ndspline.hpp"
#include "quat.hpp"

ndspline ndspline::make(const arma::mat& m) {
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

arma::mat ndspline::eval(double t) const {
    arma::mat ret(splines.size(), 1);
    for (int i = 0; i < ret.size(); ++i) {
        ret[i] = splines[i](t);
    }
    return ret;
}

arma::mat ndspline::deriv(double t) const {
    arma::mat ret(splines.size(), 1);
    for (int i = 0; i < ret.size(); ++i) {
        ret[i] = splines[i].deriv(1, t);
    }
    return ret;
}

arma::vec4 ndspline::rderiv_numeric(double t) const {
    arma::vec4 i_l = arma::normalise(eval(t));
    arma::vec4 i_r = arma::normalise(eval(t + 1e-7));
    arma::vec4 ret = quat_prod(quat_conj(i_l), i_r) / 1e-7;
    ret[0] = 0;
    return ret;
}

arma::vec4 ndspline::rderiv(double t) const {
    arma::vec4 value = eval(t);
    double norm = arma::norm(value);
    return (quat_prod(quat_conj(value), deriv(t))) / (norm * norm) * 2;
}