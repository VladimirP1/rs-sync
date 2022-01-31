#include <iostream>

#include <Eigen/Eigen>

template <class T, int D>
struct Polynomial {
    using MatT = Eigen::Matrix<T, D + 1, 1>;

    Polynomial() { Eigen::Map<MatT>{k} = MatT::Zero(); }

    Polynomial(double scalar) {
        Eigen::Map<MatT>{k} = MatT::Zero();
        k[0] = scalar;
    }

    Polynomial(const MatT& m) { Eigen::Map<MatT>{k} = m; }

    Polynomial(const std::initializer_list<T>& lst) {
        Eigen::Map<MatT>{k} = MatT::Zero();
        std::copy(lst.begin(), lst.end(), k);
    }

    Polynomial operator*(const Polynomial& other) const {
        if (Degree() + other.Degree() > D) abort();

        MatT ret = MatT::Zero();
        for (int p = 0; p <= D; ++p) {
            for (int i = 0; i <= p; ++i) {
                ret[p] += k[i] * other.k[p - i];
            }
        }
        return ret;
    }

    bool operator==(const Polynomial& other) const { return MatT{k} == MatT{other.k}; }

    bool operator!=(const Polynomial& other) const { return !(*this == other); }

    Polynomial operator+(const Polynomial& other) const {
        Polynomial ret(*this);
        ret += other;
        return ret;
    }

    Polynomial operator>>(int shift) const {
        Polynomial ret(*this);
        std::copy_backward(ret.k, ret.k + D + 1 - shift, ret.k + D + 1);
        std::fill(ret.k, ret.k + shift, 0.);
        return ret;
    }

    Polynomial operator<<(int shift) const {
        Polynomial ret(*this);
        std::copy(ret.k + shift, ret.k + D + 1, ret.k);
        std::fill(ret.k + D + 1 - shift, ret.k + D + 1, 0.);
        return ret;
    }

    Polynomial operator/(Polynomial divisor) const {
        if (divisor.Degree(1e-2) < 0) abort();
        std::cout << "enter division routine" << std::endl;
        Polynomial remainder(*this), q;
        const double tol = 1e-5;
        while (true) {
            std::cout << "rem:" << remainder.coeffs().transpose()
                      << "\ndiv:" << divisor.coeffs().transpose() << std::endl;
            int shift = remainder.Degree(tol) - divisor.Degree(tol);
            if (shift < 0) break;

            Polynomial shifted = divisor >> shift;
            q.k[shift] = remainder.k[remainder.Degree(tol)] / shifted.k[shifted.Degree(tol)];
            std::cout << "shift = " << shift << " Div by " << shifted.k[shifted.Degree(tol)]
                      << " res = " << q.k[shift] << std::endl;
            shifted *= q.k[shift];
            remainder -= shifted;
        }
        std::cout << "rem:" << remainder.coeffs().transpose()
                  << "\ndiv:" << divisor.coeffs().transpose() << std::endl;

        if (remainder.Degree(1e-2) >= 0) abort();

        return q;
    }

    Polynomial operator-(const Polynomial& other) const {
        Polynomial ret(*this);
        ret -= other;
        return ret;
    }

    Polynomial& operator+=(const Polynomial& other) {
        Eigen::Map<MatT>{k} += MatT{other.k};
        return *this;
    }

    Polynomial& operator-=(const Polynomial& other) {
        Eigen::Map<MatT>{k} -= MatT{other.k};
        return *this;
    }

    Polynomial& operator*=(const Polynomial& other) {
        *this = *this * other;
        return *this;
    }

    Polynomial& operator/=(const Polynomial& other) {
        *this = *this / other;
        return *this;
    }

    Polynomial operator-() const { return {-MatT{k}}; }

    MatT coeffs() const { return MatT{k}; }

    int Degree(double tol = 1e-5) const {
        for (int i = D + 1; i--;) {
            if (fabs(k[i]) > tol) {
                return i;
            }
        }
        return -1;
    }

   private:
    T k[D + 1];
};

// From
// https://cs.stackexchange.com/questions/124759/determinant-calculation-bareiss-vs-gauss-algorithm
template <int dim, int deg>
Polynomial<double, deg> PolyDet(Eigen::Matrix<Polynomial<double, deg>, dim, dim> matrix) {
    static constexpr double tol = 1e-3;
    if (dim <= 0) {
        return {};
    }
    double sign = 1;
    for (int k = 0; k < dim - 1; k++) {
        // Pivot - row swap needed
        if (matrix(k, k).Degree(tol) < 0) {
            int m = 0;
            for (m = k + 1; m < dim; m++) {
                if (matrix(m, k).Degree(tol) >= 0) {
                    auto tmp = matrix.template block<1, dim>(m, 0);
                    matrix.template block<1, dim>(m, 0) = matrix.template block<1, dim>(k, 0);
                    matrix.template block<1, dim>(k, 0) = tmp;
                    sign = -sign;
                    break;
                }
            }
            // No entries != 0 found in column k -> det = 0
            if (m == dim) {
                return {};
            }
        }
        // Apply formula
        for (int i = k + 1; i < dim; i++) {
            for (int j = k + 1; j < dim; j++) {
                matrix(i, j) = matrix(k, k) * matrix(i, j) - matrix(i, k) * matrix(k, j);
                if (k != 0) {
                    matrix(i, j) /= matrix(k - 1, k - 1);
                }
            }
        }
    }

    return matrix(dim - 1, dim - 1) * sign;
}

int main() {
    Polynomial<double, 6> p({1, 0, 1, 0, 0, 0});
    Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Random();
    Eigen::Matrix<double, 9, 9> B = Eigen::Matrix<double, 9, 9>::Random();

    constexpr int deg = 18;

    Eigen::Matrix<Polynomial<double, deg>, 9, 9> q =
        A.cast<Polynomial<double, deg>>() * Polynomial<double, deg>({0, 1}) +
        B.cast<Polynomial<double, deg>>();
    std::cout << PolyDet(q).coeffs() << std::endl;

    // Polynomial<double, 60> a{.6, .2, .3, .7, 0, 0};
    // Polynomial<double, 60> b{1, .3, 0, 0, .9, 0};
    // auto c = (a * b) / b;
    // std::cout << c.coeffs().transpose() << " | " << a.coeffs().transpose() << std::endl;
    return 0;
}