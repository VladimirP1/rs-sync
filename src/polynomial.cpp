#include <iostream>

#include <gmpxx.h>

#include <Eigen/Eigen>

#include <math/polynomial.hpp>

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
    W << 0, 1, 0, -1, 0, 0, 0, 0, 1;
    R1 = U * W * Vt;
    R2 = U * W.transpose() * Vt;
    t = U.col(2);
}

int main() {
    Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Zero(),
                                B = Eigen::Matrix<double, 9, 9>::Zero();

    std::vector<Eigen::Vector3d> p0s, p1s;
    for (int i = 0; i < 9; ++i) {
        Eigen::Vector2d ts = Eigen::Vector2d::Random();
        Eigen::Vector2d p0 = Eigen::Vector2d::Random();
        Eigen::Vector2d v = Eigen::Vector2d::Ones();
        Eigen::Vector2d p1 = p0 + v * .01;
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