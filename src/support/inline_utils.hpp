#pragma once

#include <armadillo>

inline arma::mat safe_normalize(arma::mat m) {
    double norm = arma::norm(m);
    if (norm < 1e-8) {
        return m;
    }
    return m / norm;
}

inline int mtrand(const int& min, const int& max) {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}