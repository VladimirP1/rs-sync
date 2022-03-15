#include <iostream>
#include <armadillo>
#include <functional>
#include <gsl/gsl_multimin.h>

namespace arma {
arma::vec wrap(gsl_vector *v) { return arma::vec(v->data, v->size, false, true); }
const arma::vec wrap(const gsl_vector *v) { return arma::vec(v->data, v->size, false, true); }
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

    static double _f(const gsl_vector *v, void *params) {
        auto _this = static_cast<MultiminFunction *>(params);
        return _this->f_(arma::wrap(v));
    }

    static void _df(const gsl_vector *v, void *params, gsl_vector *df) {
        auto _this = static_cast<MultiminFunction *>(params);
        arma::wrap(df) = std::get<1>(_this->fdf_(arma::wrap(v)));
    }

    static void _fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df) {
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

    MultiminMinimizer(gsl_multimin_fdfminimizer_type *type, size_t n) {
        minimizer = gsl_multimin_fdfminimizer_alloc(type, n);
    }

    ~MultiminMinimizer() { gsl_multimin_fdfminimizer_free(minimizer); }

    gsl_multimin_fdfminimizer *minimizer;
};
}  // namespace Gsl

int main() {
    Gsl::MultiminFunction mf(3);
    mf.SetF([](arma::vec v) { return arma::accu(v % v); });
    mf.SetFdF([](arma::vec v) { return std::make_tuple(arma::accu(v % v), (2 * v).eval()); });

    gsl_vector *x = gsl_vector_alloc(3);
    arma::wrap(x) = {1, 1, 1};

    gsl_multimin_fdfminimizer *minimizer =
        gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_vector_bfgs2, 3);

    gsl_multimin_fdfminimizer_set(minimizer, &mf.gsl_func, x, .5, 1e-9);

    for (int i = 0; i < 10; i++) {
        std::cout << arma::wrap(minimizer->x).t() << std::endl;
        gsl_multimin_fdfminimizer_iterate(minimizer);
    }

    gsl_multimin_fdfminimizer_free(minimizer);
    gsl_vector_free(x);
}