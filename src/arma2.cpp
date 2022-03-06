#include <iostream>

#include <armadillo>

static std::tuple<arma::mat, arma::mat> sqr_jac(arma::mat x) { return {x % x, arma::diagmat(2. * x)}; }

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

static std::tuple<arma::mat, arma::mat, arma::mat> div_jac(arma::mat x, arma::mat y) {
    if (y.n_cols != 1 || y.n_rows != 1) throw std::runtime_error("size mismatch");
    arma::mat dx(x.n_rows, x.n_rows);
    dx.eye();
    return {x / y[0], dx / y[0], -x / (y[0] * y[0])};
}

void print_shape(arma::mat x) { std::cout << x.n_rows << " x " << x.n_cols << std::endl; }

int main() {
    arma::mat x(10, 1);
    arma::mat y(1, 1);

    x.ones();
    y.ones();

    // x*=1;
    // y*=1;
    
    {
        auto [a, b, c] = div_jac(x, y);

        print_shape(a);
        print_shape(b);
        print_shape(c);
    }

    std::cout << std::endl;
    {
        auto [a, b] = sum_jac(x);

        print_shape(a);
        print_shape(b);
    }

    std::cout << std::endl;
    {
        auto [a, b] = log1p_jac(x);

        print_shape(a);
        print_shape(b);
    }

    std::cout << std::endl;
    {
        auto [a, b] = sqrt_jac(x);

        print_shape(a);
        print_shape(b);
    }

    std::cout << std::endl;
    {
        auto [a, b] = sqr_jac(x);

        print_shape(a);
        print_shape(b);
    }
}