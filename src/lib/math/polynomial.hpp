#pragma once

#include <Eigen/Eigen>

template <class T, int D>
struct Polynomial {
    using MatT = Eigen::Matrix<T, D + 1, 1>;

    Polynomial() { std::fill(k, k + D + 1, T{0}); }

    Polynomial(T scalar) {
        std::fill(k, k + D + 1, T{0});
        k[0] = scalar;
    }

    Polynomial(const MatT& m) { Eigen::Map<MatT>{k} = m; }

    Polynomial(const std::initializer_list<T>& lst) {
        std::fill(k, k + D + 1, T{0});
        std::copy(lst.begin(), lst.end(), k);
    }

    Polynomial operator*(const Polynomial& other) const {
        if (Degree() + other.Degree() > D) {
            abort();
        }

        Polynomial prod;
        for (int p = 0; p <= D; ++p) {
            for (int i = 0; i <= p; ++i) {
                prod.k[p] += k[i] * other.k[p - i];
                if (!std::isfinite(prod.k[p])) abort();
            }
        }

        return prod;
    }

    bool operator==(const Polynomial& other) const {
        for (int i = 0; i < D + 1; ++i) {
            if (k[i] != other.k[i]) return false;
        }
        return true;
    }

    bool operator!=(const Polynomial& other) const { return !(*this == other); }

    Polynomial operator+(const Polynomial& other) const {
        Polynomial ret(*this);
        ret += other;
        return ret;
    }

    Polynomial operator>>(int shift) const {
        Polynomial ret(*this);
        std::copy_backward(ret.k, ret.k + D + 1 - shift, ret.k + D + 1);
        std::fill(ret.k, ret.k + shift, T{0});
        return ret;
    }

    Polynomial operator<<(int shift) const {
        Polynomial ret(*this);
        std::copy(ret.k + shift, ret.k + D + 1, ret.k);
        std::fill(ret.k + D + 1 - shift, ret.k + D + 1, T{0});
        return ret;
    }

    Polynomial operator/(Polynomial divisor) const {
        if (divisor.Degree() < 0) abort();

        Polynomial remainder(*this), q;
        while (true) {
            int rd = remainder.Degree(), dd = divisor.Degree(), shift = rd - dd;
            if (rd - dd < 0) break;
            Polynomial shifted = divisor >> shift;
            q.k[shift] = remainder.k[rd] / shifted.k[dd + shift];
            if (!std::isfinite(q.k[shift])) abort();
            shifted *= q.k[shift];
            remainder -= shifted;
        }

        // This check only works properly when working with bignum rationals
        // if (remainder.Degree() >= 0) {abort();}

        return q;
    }

    Polynomial operator-(const Polynomial& other) const {
        Polynomial ret(*this);
        ret -= other;
        return ret;
    }

    Polynomial& operator+=(const Polynomial& other) {
        for (int i = 0; i < D + 1; ++i) k[i] += other.k[i];

        return *this;
    }

    Polynomial& operator-=(const Polynomial& other) {
        for (int i = 0; i < D + 1; ++i) k[i] -= other.k[i];
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

    Polynomial operator-() const {
        Polynomial ret{*this};
        for (int i = 0; i < D + 1; ++i) ret.k[i] -= -ret.k[i];
        return ret;
    }

    MatT coeffs() const { return MatT{k}; }

    int Degree(double eps = 1e-12) const {
        for (int i = D + 1; i--;) {
            if (!(fabs(k[i]) < eps)) {
                return i;
            }
        }
        return -1;
    }

    T k[D + 1];
};

// From
// https://cs.stackexchange.com/questions/124759/determinant-calculation-bareiss-vs-gauss-algorithm
template <int dim, int deg, class T>
Polynomial<T, deg> RatPolyDet(Eigen::Matrix<Polynomial<T, deg>, dim, dim> matrix) {
    if (dim <= 0) {
        return {};
    }
    T sign = T{1};
    for (int k = 0; k < dim - 1; k++) {
        // Pivot - row swap needed
        if (matrix(k, k).Degree() < 0) {
            int m = 0;
            for (m = k + 1; m < dim; m++) {
                if (matrix(m, k).Degree() >= 0) {
                    matrix.row(m).swap(matrix.row(k));
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