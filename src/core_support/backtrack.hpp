#pragma once

#include <armadillo>

class Backtrack {
    struct backtrack_hyper {
        double sufficent_decrease{.7};
        double decay{.1};
        double initial_step{1};
        int max_iterations{20};
    };

    backtrack_hyper hyper;
    std::function<std::tuple<double, arma::vec>(arma::vec)> f_and_grad;
    std::function<double(arma::vec)> f_only;

   public:
    void SetHyper(double sufficent_decrease, double decay, double initial_step, int max_iterations);

    void SetObjective(std::function<double(arma::vec)> f_only);

    void SetObjective(std::function<std::tuple<double, arma::vec>(arma::vec)> f_and_grad);

    arma::vec Step(arma::vec x0);
};