#pragma once
#include <armadillo>
#include <functional>
#include <gsl/gsl_multimin.h>
#include <iostream>

namespace arma {
inline arma::vec wrap(gsl_vector *v) { return arma::vec(v->data, v->size, false, true); }
inline const arma::vec wrap(const gsl_vector *v) {
    return arma::vec(v->data, v->size, false, true);
}
}  // namespace arma

namespace Gsl {

struct MultiminFunction {
    MultiminFunction(const MultiminFunction &) = delete;
    MultiminFunction(MultiminFunction &&) = delete;
    MultiminFunction &operator=(const MultiminFunction &) = delete;
    MultiminFunction &operator=(MultiminFunction &&) = delete;
    MultiminFunction(int n) {
        gsl_func.f = &_f;
        gsl_func.df = &_df;
        gsl_func.fdf = &_fdf;
        gsl_func.n = n;
        gsl_func.params = static_cast<void *>(this);
    }

    void SetF(std::function<double(arma::vec)> f) { f_ = std::move(f); }

    void SetFdF(std::function<std::tuple<double, arma::vec>(arma::vec)> fdf) {
        fdf_ = std::move(fdf);
    }

    gsl_multimin_function_fdf gsl_func;

   private:
    std::function<double(arma::vec)> f_;
    std::function<std::tuple<double, arma::vec>(arma::vec)> fdf_;

    static inline double _f(const gsl_vector *v, void *params) {
        auto _this = static_cast<MultiminFunction *>(params);
        return _this->f_(arma::wrap(v));
    }

    static inline void _df(const gsl_vector *v, void *params, gsl_vector *df) {
        auto _this = static_cast<MultiminFunction *>(params);
        arma::wrap(df).zeros();
        // arma::wrap(df) = std::get<1>(_this->fdf_(arma::wrap(v)));
        std::cout << std::get<0>(_this->fdf_(arma::wrap(v))) << std::endl << arma::wrap(v).t() << arma::wrap(df).t() << std::endl;
    }

    static inline void _fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df) {
        auto _this = static_cast<MultiminFunction *>(params);
        *f = _this->f_(arma::wrap(x));
        arma::wrap(df) = std::get<1>(_this->fdf_(arma::wrap(x)));
    }
};

struct MultiminMinimizer {
    MultiminMinimizer(const MultiminMinimizer &) = delete;
    MultiminMinimizer(MultiminMinimizer &&) = delete;
    MultiminMinimizer &operator=(const MultiminMinimizer &) = delete;
    MultiminMinimizer &operator=(MultiminMinimizer &&) = delete;

    MultiminMinimizer(const gsl_multimin_fdfminimizer_type *type, size_t n) {
        minimizer = gsl_multimin_fdfminimizer_alloc(type, n);
    }

    ~MultiminMinimizer() { gsl_multimin_fdfminimizer_free(minimizer); }

    gsl_multimin_fdfminimizer *minimizer;
};
}  // namespace Gsl
