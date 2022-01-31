#include <iostream>

#include <gmpxx.h>

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
            std::cout << coeffs() << "\n\\\n"
                      << other.coeffs() << "\n\\\n"
                      << other.Degree() << std::endl;
            abort();
        }

        Polynomial prod;
        for (int p = 0; p <= D; ++p) {
            for (int i = 0; i <= p; ++i) {
                prod.k[p] += k[i] * other.k[p - i];
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

        // std::cout << "enter division routine" << std::endl;
        Polynomial remainder(*this), q;
        int target_degree = remainder.Degree() - divisor.Degree();
        while (true) {
            // std::cout << "rem:" << remainder.coeffs().transpose()
            //           << "\ndiv:" << divisor.coeffs().transpose() << std::endl;
            int shift = remainder.Degree() - divisor.Degree();
            if (shift < 0) break;

            Polynomial shifted = divisor >> shift;
            q.k[shift] = remainder.k[remainder.Degree()] / shifted.k[shifted.Degree()];
            // std::cout << "shift = " << shift << " " << q.Degree() << " Div by " <<
            // shifted.k[shifted.Degree()]
            //           << " res = " << q.k[shift] << std::endl;
            shifted *= q.k[shift];
            remainder -= shifted;
        }
        // std::cout << "rem:" << remainder.coeffs().transpose()
        //           << "\ndiv:" << divisor.coeffs().transpose() << std::endl;

        if (remainder.Degree() >= 0) abort();

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

    int Degree() const {
        for (int i = D + 1; i--;) {
            if (!(k[i] == 0)) {
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

Eigen::Matrix<Polynomial<mpq_class, 18>, 9, 9> MergeAB(Eigen::Matrix<double, 9, 9> A,
                                                       Eigen::Matrix<double, 9, 9> B) {
    using P = Polynomial<mpq_class, 18>;
    return B.cast<P>() + (A.cast<P>() * P{0, 1});
}

void SetRow(int row, Eigen::Matrix<double, 9, 9>& A, Eigen::Matrix<double, 9, 9>& B,
            Eigen::Vector2d point1, Eigen::Vector2d point2, double t1, double t2) {
    double l1 = t2 - t1;
    double l2 = l1 * (t2 + t1 + 1.) / 2.;
    double x1 = point1[0], y1 = point1[1];
    double x2 = point2[0], y2 = point2[1];

    std::cout << "l1=" << l1 << " l2=" << l2 << std::endl;
    std::cout << "x1=" << x1 << " y1=" << y1 << std::endl;
    std::cout << "x2=" << x2 << " y2=" << y2 << std::endl;

    // clang-format off
    B.block<1,9>(row,0) << 
        l1 * x1 * x1 - x1 * x1 + x1 * x2,
        l1 * x1 * y1 - x1 * y1 + x2 * y1, 
        l1 * x1 - x1 + x2,
        l1 * x1 * y1 - x1 * y1 + x1 * y2,
        l1 * y1 * y1 - y1 * y1 + y1 * y2, 
        l1 * y1 - y1 + y2, 
        x1,
        y1, 
        1;
    A.block<1,9>(row,0) << 
         l2 * x1 * x1,
         l2 * x1 * y1, 
         l2 * x1,
         l2 * x1 * y1,
         l2 * y1 * y1, 
         l2 * y1, 
         0,
         0, 
         0;
    // clang-format on
}

void DecomposeEssential(Eigen::Matrix3d E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2,
                        Eigen::Vector3d& t) {
    {
        Eigen::JacobiSVD<decltype(E)> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d D = svd.singularValues().asDiagonal();
        D(2, 2) = 0;
        E = svd.matrixU() * D * svd.matrixV().transpose();
    }

    Eigen::JacobiSVD<decltype(E)> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix<double, 3, 3> W, U = svd.matrixU(), Vt = svd.matrixV().transpose();

    if (U.determinant() < 0) U *= -1.;
    if (Vt.determinant() < 0) Vt *= -1;

    // clang-format off
    W <<
         0, 1, 0,
        -1, 0, 0,
         0, 0, 1;
    // clang-format on

    R1 = U * W * Vt;
    R2 = U * W.transpose() * Vt;
    t = U.col(2);
}

int main() {
    Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Zero(),
                                B = Eigen::Matrix<double, 9, 9>::Zero();

    std::vector<Eigen::Vector3d> p0s, p1s;
    Eigen::Matrix<double, 9, 2> pts;
    pts << -1, -1, 1, -1, 1, 1, -1, 1, 0, 1, 0, -1, 1, 0, -1, 0, 0, 0;
    for (int i = 0; i < 9; ++i) {
        Eigen::Vector2d ts = Eigen::Vector2d::Random();
        Eigen::Vector2d p0 = pts.row(i);
        Eigen::Vector2d v = Eigen::Vector2d::Ones();
        Eigen::Vector2d p1 = p0 + v *.01;
        p0s.emplace_back();
        p0s.back() << p0.x(), p0.y(), 1.;
        p1s.emplace_back();
        p1s.back() << p1.x(), p1.y(), 1.;
        SetRow(i, A, B, p0, p1, 0, 1);
    }
    Eigen::Matrix<Polynomial<mpq_class, 18>, 9, 9> PolyMat = MergeAB(A, B);

    auto det = RatPolyDet(PolyMat);
    det /= det.k[det.Degree()];

    int deg = det.Degree();

    // std::cout << "Polynomial cooficients: ";
    // for (int i = 0; i < det.Degree(); ++i) {
    //     std::cout << det.k[i].get_d() << " ";
    // }
    // std::cout << std::endl << std::endl;

    double C_solution{};
    if (deg > 0) {
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(deg, deg);

        for (int j = 0; j < deg; ++j) {
            if (j + 1 < deg) C(j + 1, j) = 1;
            C(j, deg - 1) = -det.k[j].get_d();
        }

        C_solution = Eigen::JacobiSVD<decltype(C)>(C).singularValues().tail(1)[0];

        // std::cout << "Companion matrix:" << std::endl;
        // std::cout << C << std::endl << std::endl;
        // std::cout << "Min eigenvalue of C: " << C_solution << std::endl;
    }

    Eigen::Matrix<double, 9, 9> D = B + A * C_solution;


    Eigen::JacobiSVD<decltype(D)> solver(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> E(solver.matrixV().col(8).real().data());
    // clang-format off
    // E << 
    //     E_col(0,0), E_col(0,1), E_col(0,2),
    //     E_col(0,3), E_col(0,4), E_col(0,5),
    //     E_col(0,6), E_col(0,7), E_col(0,8);
    // clang-format on

    // std::cout << "Determinant of resulting matrix: " << D.determinant() << std::endl << std::endl;
    // std::cout << "Essential matrix:" << std::endl;
    // std::cout << E << std::endl;
    // std::cout << "its determinant is: " << E.determinant() << std::endl << std::endl;
    std::cout << "Errors: " << std::endl;
    for (int i = 0; i < 9; ++i) {
        std::cout << p1s[i].transpose() * E * p0s[i] << " ";
    }
    std::cout << std::endl << std::endl;



    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    DecomposeEssential(E, R1, R2, t);
    std::cout << "R1:\n" << R1 << "\n\nR2:\n" << R2 << "\n\nt:\n" << t << std::endl;
    Eigen::AngleAxis<double> r1, r2;
    r1.fromRotationMatrix(R1);
    r2.fromRotationMatrix(R2);
    std::cout << "R1: " << r1.angle() << "  R2: " << r2.angle() << std::endl;

    return 0;
}