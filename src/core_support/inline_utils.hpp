#pragma once

#include <armadillo>

inline arma::mat safe_normalize(arma::mat m) {
    double norm = arma::norm(m);
    if (norm < 1e-12) {
        return m;
    }
    return m / norm;
}

inline int mtrand(const int& min, const int& max) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

inline std::tuple<arma::mat, arma::mat> sqr_jac(arma::mat x) {
    return {x % x, arma::diagmat(2. * x)};
}

inline std::tuple<arma::mat, arma::mat> sqrt_jac(arma::mat x) {
    x = arma::sqrt(x);
    return {x, arma::diagmat(1. / (2. * x))};
}

inline std::tuple<arma::mat, arma::mat> log1p_jac(arma::mat x) {
    return {arma::log1p(x), arma::diagmat(1. / (1. + x))};
}

inline std::tuple<arma::mat, arma::mat> sum_jac(arma::mat x) {
    arma::mat d(1, x.n_rows);
    d.ones();
    return {arma::sum(x), d};
}

inline std::tuple<arma::mat, arma::mat, arma::mat> div_jac(arma::mat x, double y) {
    arma::mat dx(x.n_rows, x.n_rows);
    dx.eye();
    return {x / y, dx / y, -x / (y * y)};
}

inline std::tuple<arma::mat, arma::mat> mul_const_jac(arma::mat x, double y) {
    arma::mat dx(x.n_rows, x.n_rows);
    dx.eye();
    return {x * y, dx * y};
}

inline double clamp_k(double k) { return std::clamp(k, 1e1, 1e3); }