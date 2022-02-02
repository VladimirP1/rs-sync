#include <Eigen/Eigen>
#include <rtl/RTL.hpp>

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include <iostream>

namespace {

template <class T>
void SetRow(int row, T& A, T& B, Eigen::Vector2d point1, Eigen::Vector2d point2, double t1,
            double t2) {
    double l1 = t2 - t1;
    double l2 = l1 * (t2 + t1 + 1.) / 2.;
    double x1 = point1[0], y1 = point1[1];
    double x2 = point2[0], y2 = point2[1];

    // clang-format off
    // B.template block<1,9>(row,0) << 
    //     l1 * x1 * x1 - x1 * x1 + x1 * x2,
    //     l1 * x1 * y1 - x1 * y1 + x2 * y1, 
    //     l1 * x1 - x1 + x2,
    //     l1 * x1 * y1 - x1 * y1 + x1 * y2,
    //     l1 * y1 * y1 - y1 * y1 + y1 * y2, 
    //     l1 * y1 - y1 + y2, 
    //     x1,
    //     y1, 
    //     1;
    B.template block<1,9>(row,0) << 
        l1*x1*(l1*l1*x1 - l1*x1 + x2),
        l1*y1*(l1*l1*x1 - l1*x1 + x2),
        l1*l1*x1 - l1*x1 + x2,
        l1*x1*(l1*l1*y1 - l1*y1 + y2),
        l1*y1*(l1*l1*y1 - l1*y1 + y2),
        l1*l1*y1 - l1*y1 + y2,
        l1*x1,
        l1*y1,
        1;
    // A.template block<1,9>(row,0) << 
    //      l2 * x1 * x1,
    //      l2 * x1 * y1, 
    //      l2 * x1,
    //      l2 * x1 * y1,
    //      l2 * y1 * y1, 
    //      l2 * y1, 
    //      0,
    //      0, 
    //      0;
    A.template block<1,9>(row,0) << 
        3*l1*l1*l2*x1*x1 - 2*l1*l2*x1*x1 + l2*x1*x2,
        3*l1*l1*l2*x1*y1 - 2*l1*l2*x1*y1 + l2*x2*y1,
        2*l1*l2*x1 - l2*x1,
        3*l1*l1*l2*x1*y1 - 2*l1*l2*x1*y1 + l2*x1*y2,
        3*l1*l1*l2*y1*y1 - 2*l1*l2*y1*y1 + l2*y1*y2,
        2*l1*l2*y1 - l2*y1,
        l2*x1,
        l2*y1,
        0;
    // clang-format on
}

Eigen::Matrix<Eigen::Matrix<double, 1, 2>, 9, 9> MergeAB(Eigen::Matrix<double, 9, 9> A,
                                                         Eigen::Matrix<double, 9, 9> B) {
    Eigen::Matrix<Eigen::Matrix<double, 1, 2>, 9, 9> ret;
    for (int i = 0; i < 9 * 9; ++i) {
        ret.data()[i] << B.data()[i], A.data()[i];
    }
    return ret;
}

double SolveForK(Eigen::Matrix<double, 9, 9> A, Eigen::Matrix<double, 9, 9> B) {
    if (fabs(B.determinant()) < 1e-15) {
        return 0.;
    }

    int max_real_i = 0;
    Eigen::Matrix<std::complex<double>, 9, 1> eigens = (A * B.inverse()).eigenvalues();
    for (int i = 0; i < 9; ++i) {
        if (fabs(eigens(max_real_i, 0).imag()) > fabs(eigens(i, 0).imag())) {
            max_real_i = i;
        }
        if (fabs(eigens(max_real_i, 0).real()) < 1e-9 &&
            (eigens(max_real_i, 0).real()) > (eigens(i, 0).real())) {
            max_real_i = i;
        }
    }

    double solution = - 1. / eigens(max_real_i, 0).real();
    // std::cout << eigens.real() << "\n---------- " << solution << std::endl;
    // std::cout << solution << std::endl;

    return solution;
}

template <class T>
void NormailzePointSet(T& points, Eigen::Matrix3d& reverse) {
    reverse = Eigen::Matrix3d::Identity();
    return;

    // double scale = 0;
    // Eigen::Vector2d center = Eigen::Vector2d::Zero();

    // for (int i = 0; i < points.size(); ++i) {
    //     center += points[i].template block<2, 1>(0, 0);
    // }
    // center /= points.size();

    // for (int i = 0; i < points.size(); ++i) {
    //     scale += (points[i].template block<2, 1>(0, 0) - center).norm();
    // }
    // scale /= points.size();

    // scale = sqrt(2) / scale;

    // for (int i = 0; i < points.size(); ++i) {
    //     points[i].template block<2, 1>(0, 0) = (Eigen::Vector2d{points[i].data()} - center) * scale;
    // }

    // reverse << scale, 0, -scale * center[0], 0, scale, -scale * center[1], 0, 0, 1;
}

template <class T>
double ComputeError(const T& point1, const T& point2, std::pair<Eigen::Matrix3d, double> model) {
    Eigen::Vector3d p1, p2, Ep1, Ep2;
    double l1 = point2[2] - point1[2];
    double l2 = l1 * (point2[2] + point1[2] + 1.) / 2.;
    double beta = l1 + model.second * l2;
    // x1 * (l1 + k*l2)  + (x2 - x1)
    p1 << point1[0] * beta, point1[1] * beta, 1;
    p2 << point1[0] * beta + point2[0] - point1[0], point1[1] * beta + point2[1] - point1[1], 1;
    Ep1 = model.first * p1;
    Ep2 = model.first.transpose() * p2;

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

    // std::cout << A << std::endl << std::endl << B << std::endl << std::endl;

    double k = SolveForK(A, B);

    // k = 0;

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

    {
        Eigen::JacobiSVD<decltype(E)> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d D = svd.singularValues().asDiagonal();
        D(2, 2) = 0;
        E = svd.matrixU() * D * svd.matrixV().transpose();
    }

    return T2.transpose() * E * T1;
}

template <class T>
Eigen::Matrix3d FindEssentialMatN(T points1, T points2, double k) {
    assert(points1.size() == points2.size());

    Eigen::Matrix3d T1, T2;
    NormailzePointSet(points1, T1);
    NormailzePointSet(points2, T2);

    Eigen::MatrixXd A(points1.size(), 9), B(points1.size(), 9);

    for (size_t i = 0; i < points1.size(); ++i) {
        const auto& p1 = points1[i].template block<2, 1>(0, 0);
        const auto& p2 = points2[i].template block<2, 1>(0, 0);
        SetRow(i, A, B, p1, p2, points1[i][2], points2[i][2]);
    }

    Eigen::MatrixXd D = B + A * k;

    Eigen::JacobiSVD<decltype(D)> solver(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> E(solver.matrixV().col(8).real().data());

    {
        Eigen::JacobiSVD<decltype(E)> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d D = svd.singularValues().asDiagonal();
        D(2, 2) = 0;
        E = svd.matrixU() * D * svd.matrixV().transpose();
    }

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
        return ::ComputeError(datum.first, datum.second, model);
    }
};
}  // namespace
Eigen::Matrix3d FindEssentialMat(const std::vector<Eigen::Vector3d>& points1,
                                 const std::vector<Eigen::Vector3d>& points2,
                                 std::vector<unsigned char>& mask, double threshold, int iters,
                                 double* k_out = nullptr) {
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> points;

    if (mask.empty()) {
        mask.resize(points1.size());
        std::fill(mask.begin(), mask.end(), 1);
    }

    for (int i = 0; i < points1.size(); ++i) {
        if (mask[i]) {
            points.push_back({points1[i], points2[i]});
        }
    }

    EmEstimator estimator;
    std::pair<Eigen::Matrix3d, double> model;
    RTL::RANSAC<std::pair<Eigen::Matrix3d, double>, std::pair<Eigen::Vector3d, Eigen::Vector3d>,
                std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>>
        ransac(&estimator);

    ransac.SetParamIteration(iters);
    ransac.SetParamThreshold(threshold);
    ransac.FindBest(model, points, points.size(), 9);

    auto inlier_idxs = ransac.FindInliers(model, points, points.size());
    std::fill(mask.begin(), mask.end(), 0);
    {  // Refine model
        std::vector<Eigen::Vector3d> points1, points2;
        for (auto sample : inlier_idxs) {
            points1.push_back(points[sample].first);
            points2.push_back(points[sample].second);
            mask[sample] = 1;
        }
        model.first = FindEssentialMatN(points1, points2, model.second);
    }

    if (k_out) {
        *k_out = model.second;
    }

    return model.first;
}