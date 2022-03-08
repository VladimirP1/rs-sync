#include <iostream>

#include <armadillo>

static std::tuple<arma::mat, arma::mat> sqr_jac(arma::mat x) {
    return {x % x, arma::diagmat(2. * x)};
}

static std::tuple<arma::mat, arma::mat> sqrt_jac(arma::mat x) {
    x = arma::sqrt(x);
    return {x, arma::diagmat(1. / (2. * x))};
}

static std::tuple<arma::mat, arma::mat> log1p_jac(arma::mat x) {
    return {arma::log1p(x), arma::diagmat(1. / (1. + x))};
}

static std::tuple<arma::mat, arma::mat> sum_jac(arma::mat x) {
    arma::mat d(1, x.n_rows);
    d.ones();
    return {arma::sum(x), d};
}

static std::tuple<arma::mat, arma::mat, arma::mat> div_jac(arma::mat x, double y) {
    arma::mat dx(x.n_rows, x.n_rows);
    dx.eye();
    return {x / y, dx / y, -x / (y * y)};
}

void print_shape(arma::mat x) { std::cout << x.n_rows << " x " << x.n_cols << std::endl; }

double calc(arma::mat P, arma::mat M, double k) {
    auto [v1, j1] = std::make_tuple(P * M, P);
    auto [v2, j2] = sqr_jac(v1);

    auto [v3, j3] = sqr_jac(M);
    auto [v4, j4] = sum_jac(v3);
    auto [v5, j5, _] = div_jac(v4, k * k);

    auto [v6, j6a, j6b] = div_jac(v2, v5[0]);
    auto [v7, j7] = log1p_jac(v6);
    auto [v8, j8] = sum_jac(v7);

    return v8[0];
}

int main() {

    arma::mat P(100, 3);
    arma::mat M(3, 1);
    double k = 1e3;

    M.randn();
    P.randn();

    auto [v1, j1] = std::make_tuple(P * M, P);
    auto [v2, j2] = sqr_jac(v1);

    auto [v3, j3] = sqr_jac(M);
    auto [v4, j4] = sum_jac(v3);
    auto [v5, j5, _] = div_jac(v4, k * k);

    auto [v6, j6a, j6b] = div_jac(v2, v5[0]);
    auto [v7, j7] = log1p_jac(v6);
    auto [v8, j8] = sum_jac(v7);

    std::cout << j8 * j7 * (j6a * j2 * j1 + j6b * j5 * j4 * j3) << std::endl;

    double v = calc(P,M,k);
    {
        M[0] += 1e-7;
        double v2 = calc(P,M,k);
        std::cout << (v2 - v) / 1e-7 << std::endl;
    }
}