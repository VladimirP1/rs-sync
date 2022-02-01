#include <vector>

#include <Eigen/Eigen>
#include <gmpxx.h>

#include <rtl/RTL.hpp>

#include <math/polynomial.hpp>

#include <iostream>

void SetRow(int row, Eigen::Matrix<double, 9, 9>& A, Eigen::Matrix<double, 9, 9>& B,
            Eigen::Vector2d point1, Eigen::Vector2d point2, double t1, double t2) {
    double l1 = t2 - t1;
    double l2 = l1 * (t2 + t1 + 1.) / 2.;
    double x1 = point1[0], y1 = point1[1];
    double x2 = point2[0], y2 = point2[1];

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

Eigen::Matrix<Polynomial<double, 20>, 9, 9> MergeAB(Eigen::Matrix<double, 9, 9> A,
                                                       Eigen::Matrix<double, 9, 9> B) {
    using P = Polynomial<double, 20>;
    return B.cast<P>() + (A.cast<P>() * P{0, 1});
}

double SolveForK(Eigen::Matrix<double, 9, 9> A, Eigen::Matrix<double, 9, 9> B) {
    auto P = MergeAB(A, B);
    auto det_P = RatPolyDet(P);
    int deg_P = det_P.Degree();
    if (deg_P <= 0) {
        return 0.;
    }

    det_P /= det_P.k[deg_P];

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(deg_P, deg_P);

    for (int j = 0; j < deg_P; ++j) {
        if (j + 1 < deg_P) C(j + 1, j) = 1;
        C(j, deg_P - 1) = -det_P.k[j];
    }

    // std::cout << Eigen::JacobiSVD<decltype(C)>(C).singularValues().transpose() << std::endl;

    return Eigen::JacobiSVD<decltype(C)>(C).singularValues().tail(1)[0];
}

template <class T>
void NormailzePointSet(T& points, Eigen::Matrix3d& reverse) {
    double scale = 0;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();

    // Compute mean
    for (int i = 0; i < points.size(); ++i) {
        center += points[i].template block<2, 1>(0, 0);
    }
    center /= points.size();

    // Compute mean distance from mean
    for (int i = 0; i < points.size(); ++i) {
        scale += (points[i].template block<2, 1>(0, 0) - center).norm();
    }
    scale /= points.size();

    scale = sqrt(2) / scale;

    // Update points
    for (int i = 0; i < points.size(); ++i) {
        points[i].template block<2, 1>(0, 0) = (Eigen::Vector2d{points[i].data()} - center) * scale;
    }

    // Generate a reverse transformation
    reverse << scale, 0, -scale * center[0], 0, scale, -scale * center[1], 0, 0, 1;
}

template <class T>
double ComputeError(const T& point1, const T& point2, Eigen::Matrix3d E) {
    Eigen::Vector3d p1, p2, Ep1, Ep2;
    p1 << point1[0], point1[1], 1;
    p2 << point2[0], point2[1], 1;
    Ep1 = E * p1;
    Ep2 = E.transpose() * p2;

    double val = p2.transpose() * Ep1;
    return val * val / (Ep1.squaredNorm() + Ep2.squaredNorm());
}

template <class T>
Eigen::Matrix3d FindEssentialMat9(T points1, T points2, double* out_k = nullptr) {
    assert(points1.size() == 9);
    assert(points1.size() == points2.size());

    Eigen::Matrix3d T1, T2;
    NormailzePointSet(points1, T1);
    NormailzePointSet(points2, T2);

    Eigen::Matrix<double, 9, 9> A, B;

    for (size_t i = 0; i < 9; ++i) {
        const auto& p1 = points1[i].template block<2, 1>(0, 0);
        const auto& p2 = points2[i].template block<2, 1>(0, 0);
        SetRow(i, A, B, p1, p2, points1[i][2], points2[i][2]);
    }

    double k = SolveForK(A, B);

    if (out_k) {
        out_k[0] = k;
    }

    // std::cout << "k = " << k << std::endl;

    Eigen::Matrix<double, 9, 9> D = B + A * k;

    // std::cout << "det D = " << D.determinant() << std::endl;

    // Find the essential matrix
    Eigen::JacobiSVD<decltype(D)> solver(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> E(solver.matrixV().col(8).real().data());

    // std::cout << solver.matrixV() << std::endl;
    // std::cout << solver.singularValues().transpose() << std::endl;
    // std::cout << D * Eigen::Matrix<double, 9,1>(E.data()) << std::endl;

    // std::vector<double> errors(9);

    // std::cout << "Errors: " << std::endl;
    // for (int i = 0; i < 9; ++i) {
    //     std::cout << ComputeError(points1[i], points2[i], E) << " | "
    //               << points1[i].transpose() * E * points2[i] << std::endl;
    // }
    // std::cout << std::endl << std::endl;

    return T2.transpose() * E * T1;
}

class EmEstimator
    : public RTL::Estimator<std::pair<Eigen::Matrix3d, double>,
                            std::pair<Eigen::Vector3d, Eigen::Vector3d>,
                            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>> {
   public:
    virtual std::pair<Eigen::Matrix3d, double> ComputeModel(
        const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& data,
        const std::set<int>& samples) {
        std::vector<Eigen::Vector3d> points1, points2;
        for (auto sample : samples) {
            points1.push_back(data[sample].first);
            points2.push_back(data[sample].second);
        }
        double k{};
        auto E = FindEssentialMat9(points1, points2, &k);
        return {E, k};
    }

    virtual double ComputeError(const std::pair<Eigen::Matrix3d, double>& model,
                                const std::pair<Eigen::Vector3d, Eigen::Vector3d>& datum) {
        return ::ComputeError(datum.first, datum.second, model.first);
    }
};

void DecomposeEssential(Eigen::Matrix3d E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2,
                        Eigen::Vector3d& t);


int main() {
    srand((unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> points;

    Eigen::AngleAxis<double> rot(.05, Eigen::Vector3d{1,0,0});
    for (int i = 0; i < 90; ++i) {
        Eigen::Vector3d p1 = Eigen::Vector3d::Random();
        Eigen::Vector3d shift{0,.1,.0};
        Eigen::Vector3d p2 = rot * (p1 + shift);
        p1 /= p1[2];
        p2 /= p2[2];
        // p2[0] *= .8;
        // p2[1] *= .8;
        // p2[0] += .01;
        // p2[1] += .01;
        p1[2] = 0;
        p2[2] = 1;

        points.push_back({p1, p2});
    }

    EmEstimator est;
    RTL::RANSAC<std::pair<Eigen::Matrix3d, double>, std::pair<Eigen::Vector3d, Eigen::Vector3d>,
                std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>>
        ransac(&est);

    ransac.SetParamIteration(1000);
    ransac.SetParamThreshold(1e-7);
    std::pair<Eigen::Matrix3d, double> model;
    ransac.FindBest(model, points, points.size(), 9);

    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    DecomposeEssential(model.first, R1, R2, t);
    std::cout << "R1:\n" << R1 << "\n\nR2:\n" << R2 << "\n\nt:\n" << t << std::endl;
    Eigen::AngleAxis<double> r1, r2;
    r1.fromRotationMatrix(R1);
    r2.fromRotationMatrix(R2);
    std::cout << "R1: " << r1.angle() << "  R2: " << r2.angle() << std::endl;
    std::cout << "k: " << model.second << std::endl;

    return 0;
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