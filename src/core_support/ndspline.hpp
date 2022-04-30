#pragma once

#include <armadillo>

#include <memory>
#include <vector>

struct ndspline {

    ndspline();
    ~ndspline();

    static ndspline make(const arma::mat& m);
    arma::mat eval(double t) const;
    arma::mat deriv(double t) const;
    arma::vec4 rderiv(double t) const;
    arma::vec4 rderiv_numeric(double t) const;
private:
    std::shared_ptr<void> __pimpl__;
};