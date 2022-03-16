#pragma once

#include <armadillo>

class spline {
   public:
    spline(const arma::vec& y) { set_points(y); };

    double operator()(double x) const;
    double deriv(double x) const;

   private:
    void set_points(const arma::vec& y);

    arma::vec m_y, m_b, m_c, m_d;
};