#include <functional>
#include <tuple>
#include <iostream>

#include <armadillo>

arma::vec backtrack(std::function<std::tuple<double, arma::vec>(arma::vec)> f, arma::vec x0) {
    static constexpr double c = .9;
    static constexpr double tau = .5;
    static constexpr int limit = 20;

    auto [v, p] = f(x0);
    double m = arma::dot(p, p);
    double t = 20;
    for (int i = 0; i < limit; ++i) {
        auto [v1, m1] = f(x0 - t * p);
        // std::cout << v << " " << v1 << std::endl;
        if (v - v1 >= t * c * m) break;
        t *= tau;
        // if (i == limit - 1) std::cout << "armijo fail" << std::endl;
    }
    return x0 - t * p;
}

int main(int argc, char** argv) {
    // arma::vec x0{1};
    // for (int i = 0; i < 20; ++i) {
    //     x0 = backtrack([](arma::vec x){
    //         return std::make_tuple(double{cos(x[0])},arma::vec{-sin(x[0])});
    //     }, x0);
    //     std::cout << x0[0] << std::endl;
    // }
    auto f = [](arma::vec x) {
        double a = 1 - x[0];
        double b = x[0] * x[0] - x[1];
        double value = a * a + 100 * b * b;
        arma::vec d{200 * x[0] * x[0] * x[0] - 200 * x[0] * x[1] + x[0] - 1, -200 * b};
        return std::make_tuple(value, d);
    };

    // std::cout << std::get<1>(f({2,2})) << std::endl;

    arma::vec x0{2, 2};
    for (int i = 0; i < 200; ++i) {
        x0 = backtrack(f, x0);
        std::cout << x0 << std::endl;
    }
}