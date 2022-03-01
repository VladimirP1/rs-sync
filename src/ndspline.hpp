#pragma once

#include <spline.hpp>

#include <armadillo>

#include <vector>

struct ndspline {
    std::vector<tk::spline> splines;

    static ndspline make(const arma::mat& m);
    arma::mat eval(double t) const;
    arma::mat deriv(double t) const;
    arma::vec4 rderiv(double t) const;
    arma::vec4 rderiv_numeric(double t) const;
};