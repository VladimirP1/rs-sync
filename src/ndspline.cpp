#include "ndspline.hpp"
#include "quat.hpp"

#include "minispline.hpp"

#define p(x) (*static_cast<std::vector<spline>*>((x).__pimpl__.get()))

ndspline::ndspline() { __pimpl__.reset(new std::vector<spline>()); }

ndspline::~ndspline() {  }


ndspline ndspline::make(const arma::mat& m) {
    ndspline ret;
    for (int row = 0; row < m.n_rows; ++row) {
        p(ret).push_back(spline(m.row(row).t()));
    }
    return ret;
}

arma::mat ndspline::eval(double t) const {
    arma::mat ret(p(*this).size(), 1);
    for (int i = 0; i < ret.size(); ++i) {
        ret[i] = p(*this)[i](t);
    }
    return ret;
}

arma::mat ndspline::deriv(double t) const {
    arma::mat ret(p(*this).size(), 1);
    for (int i = 0; i < ret.size(); ++i) {
        ret[i] = p(*this)[i].deriv(t);
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