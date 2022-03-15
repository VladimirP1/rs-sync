#include "backtrack.hpp"

arma::vec Backtrack::Step(arma::vec x0) {
    auto [v, p] = f_and_grad(x0);
    double m = arma::dot(p, p);
    double t = hyper.initial_step;
    for (int i = 0; i < hyper.max_iterations; ++i) {
        auto v1 = f_only(x0 - t * p);
        if (v - v1 >= t * hyper.sufficent_decrease * m) break;
        t *= hyper.decay;
    }
    return -t * p;
}

void Backtrack::SetHyper(double sufficent_decrease, double decay, double initial_step,
                         int max_iterations) {
    hyper.sufficent_decrease = sufficent_decrease;
    hyper.decay = decay;
    hyper.initial_step = initial_step;
    hyper.max_iterations = max_iterations;
}

void Backtrack::SetObjective(std::function<std::tuple<double, arma::vec>(arma::vec)> f_and_grad) {
    if (!f_only) f_only = [f_and_grad](arma::vec x) { return std::get<0>(f_and_grad(x)); };
    this->f_and_grad = std::move(f_and_grad);
}

void Backtrack::SetObjective(std::function<double(arma::vec)> f_only) { this->f_only = std::move(f_only); }
