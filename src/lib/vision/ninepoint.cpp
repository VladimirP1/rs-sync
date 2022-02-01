#include <Eigen/Eigen>
#include <rtl/RTL.hpp>

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include <iostream>

namespace {
struct Det99Computer {
    Det99Computer() {
        std::array<uint8_t, 9> perm{0, 1, 2, 3, 4, 5, 6, 7, 8};

        for (int i = 0; i < 362880; ++i) {
            if (i % 6 == 0) {
                std::array<uint8_t, 6> tmp;
                std::copy_n(perm.begin(), 6, tmp.begin());
                left_tasks.push_back(tmp);
            }
            std::array<uint8_t, 3> tmp;
            std::copy_n(perm.begin() + 6, 3, tmp.begin());
            bool sign = Sign(perm.data(), 9);
            right_tasks.push_back({left_tasks.size() - 1, sign, tmp});

            std::next_permutation(perm.begin(), perm.end());
        }
    }

    Eigen::Matrix<double, 1, 7> Compute(
        const Eigen::Matrix<Eigen::Matrix<double, 1, 2>, 9, 9>& matrix) const {
        Eigen::Matrix<double, 1, 7> det = Eigen::Matrix<double, 1, 7>::Zero();
        std::vector<Eigen::Matrix<double, 1, 7>> left_results;
        left_results.reserve(60480);
        for (const auto& task : left_tasks) {
            const Eigen::Matrix<double, 1, 2> x0 = matrix(task[0], 0), x1 = matrix(task[1], 1),
                                              x2 = matrix(task[2], 2), x3 = matrix(task[3], 3),
                                              x4 = matrix(task[4], 4), x5 = matrix(task[5], 5);
            Eigen::Matrix<double, 1, 7> clr;
            clr << x0[0] * x1[0] * x2[0] * x3[0] * x4[0] * x5[0],
                x0[0] * x1[0] * x2[0] * x3[0] * x4[0] * x5[1] +
                    x0[0] * x1[0] * x2[0] * x3[0] * x4[1] * x5[0] +
                    x0[0] * x1[0] * x2[0] * x3[1] * x4[0] * x5[0] +
                    x0[0] * x1[0] * x2[1] * x3[0] * x4[0] * x5[0] +
                    x0[0] * x1[1] * x2[0] * x3[0] * x4[0] * x5[0] +
                    x0[1] * x1[0] * x2[0] * x3[0] * x4[0] * x5[0],
                x0[0] * x1[0] * x2[0] * x3[0] * x4[1] * x5[1] +
                    x0[0] * x1[0] * x2[0] * x3[1] * x4[0] * x5[1] +
                    x0[0] * x1[0] * x2[0] * x3[1] * x4[1] * x5[0] +
                    x0[0] * x1[0] * x2[1] * x3[0] * x4[0] * x5[1] +
                    x0[0] * x1[0] * x2[1] * x3[0] * x4[1] * x5[0] +
                    x0[0] * x1[0] * x2[1] * x3[1] * x4[0] * x5[0] +
                    x0[0] * x1[1] * x2[0] * x3[0] * x4[0] * x5[1] +
                    x0[0] * x1[1] * x2[0] * x3[0] * x4[1] * x5[0] +
                    x0[0] * x1[1] * x2[0] * x3[1] * x4[0] * x5[0] +
                    x0[0] * x1[1] * x2[1] * x3[0] * x4[0] * x5[0] +
                    x0[1] * x1[0] * x2[0] * x3[0] * x4[0] * x5[1] +
                    x0[1] * x1[0] * x2[0] * x3[0] * x4[1] * x5[0] +
                    x0[1] * x1[0] * x2[0] * x3[1] * x4[0] * x5[0] +
                    x0[1] * x1[0] * x2[1] * x3[0] * x4[0] * x5[0] +
                    x0[1] * x1[1] * x2[0] * x3[0] * x4[0] * x5[0],
                x0[0] * x1[0] * x2[0] * x3[1] * x4[1] * x5[1] +
                    x0[0] * x1[0] * x2[1] * x3[0] * x4[1] * x5[1] +
                    x0[0] * x1[0] * x2[1] * x3[1] * x4[0] * x5[1] +
                    x0[0] * x1[0] * x2[1] * x3[1] * x4[1] * x5[0] +
                    x0[0] * x1[1] * x2[0] * x3[0] * x4[1] * x5[1] +
                    x0[0] * x1[1] * x2[0] * x3[1] * x4[0] * x5[1] +
                    x0[0] * x1[1] * x2[0] * x3[1] * x4[1] * x5[0] +
                    x0[0] * x1[1] * x2[1] * x3[0] * x4[0] * x5[1] +
                    x0[0] * x1[1] * x2[1] * x3[0] * x4[1] * x5[0] +
                    x0[0] * x1[1] * x2[1] * x3[1] * x4[0] * x5[0] +
                    x0[1] * x1[0] * x2[0] * x3[0] * x4[1] * x5[1] +
                    x0[1] * x1[0] * x2[0] * x3[1] * x4[0] * x5[1] +
                    x0[1] * x1[0] * x2[0] * x3[1] * x4[1] * x5[0] +
                    x0[1] * x1[0] * x2[1] * x3[0] * x4[0] * x5[1] +
                    x0[1] * x1[0] * x2[1] * x3[0] * x4[1] * x5[0] +
                    x0[1] * x1[0] * x2[1] * x3[1] * x4[0] * x5[0] +
                    x0[1] * x1[1] * x2[0] * x3[0] * x4[0] * x5[1] +
                    x0[1] * x1[1] * x2[0] * x3[0] * x4[1] * x5[0] +
                    x0[1] * x1[1] * x2[0] * x3[1] * x4[0] * x5[0] +
                    x0[1] * x1[1] * x2[1] * x3[0] * x4[0] * x5[0],
                x0[0] * x1[0] * x2[1] * x3[1] * x4[1] * x5[1] +
                    x0[0] * x1[1] * x2[0] * x3[1] * x4[1] * x5[1] +
                    x0[0] * x1[1] * x2[1] * x3[0] * x4[1] * x5[1] +
                    x0[0] * x1[1] * x2[1] * x3[1] * x4[0] * x5[1] +
                    x0[0] * x1[1] * x2[1] * x3[1] * x4[1] * x5[0] +
                    x0[1] * x1[0] * x2[0] * x3[1] * x4[1] * x5[1] +
                    x0[1] * x1[0] * x2[1] * x3[0] * x4[1] * x5[1] +
                    x0[1] * x1[0] * x2[1] * x3[1] * x4[0] * x5[1] +
                    x0[1] * x1[0] * x2[1] * x3[1] * x4[1] * x5[0] +
                    x0[1] * x1[1] * x2[0] * x3[0] * x4[1] * x5[1] +
                    x0[1] * x1[1] * x2[0] * x3[1] * x4[0] * x5[1] +
                    x0[1] * x1[1] * x2[0] * x3[1] * x4[1] * x5[0] +
                    x0[1] * x1[1] * x2[1] * x3[0] * x4[0] * x5[1] +
                    x0[1] * x1[1] * x2[1] * x3[0] * x4[1] * x5[0] +
                    x0[1] * x1[1] * x2[1] * x3[1] * x4[0] * x5[0],
                x0[0] * x1[1] * x2[1] * x3[1] * x4[1] * x5[1] +
                    x0[1] * x1[0] * x2[1] * x3[1] * x4[1] * x5[1] +
                    x0[1] * x1[1] * x2[0] * x3[1] * x4[1] * x5[1] +
                    x0[1] * x1[1] * x2[1] * x3[0] * x4[1] * x5[1] +
                    x0[1] * x1[1] * x2[1] * x3[1] * x4[0] * x5[1] +
                    x0[1] * x1[1] * x2[1] * x3[1] * x4[1] * x5[0],
                x0[1] * x1[1] * x2[1] * x3[1] * x4[1] * x5[1];
            left_results.push_back(clr);
        }

        for (const auto& [left_idx, sign, task] : right_tasks) {
            double b = (sign ? -1. : 1.) * matrix(task[0], 6)[0] * matrix(task[1], 7)[0] *
                       matrix(task[2], 8)[0];
            det += left_results[left_idx] * b;
        }
        return det;
    }

   private:
    std::vector<std::array<uint8_t, 6>> left_tasks;
    // left_task + sign + right rows
    std::vector<std::tuple<int, bool, std::array<uint8_t, 3>>> right_tasks;

    static int Sign(uint8_t* a, int n) {
        int swaps = 0;
        int b[n];
        std::copy(a, a + n, b);
        for (int i = 0; i < n; ++i) {
            auto it = std::min_element(b + i, b + n);
            if (*it == b[i]) continue;
            std::swap(b[i], *it);
            ++swaps;
        }
        return swaps % 2;
    }
};

template <class T>
void SetRow(int row, T& A, T& B, Eigen::Vector2d point1, Eigen::Vector2d point2, double t1,
            double t2) {
    double l1 = t2 - t1;
    double l2 = l1 * (t2 + t1 + 1.) / 2.;
    double x1 = point1[0], y1 = point1[1];
    double x2 = point2[0], y2 = point2[1];

    // clang-format off
    B.template block<1,9>(row,0) << 
        l1 * x1 * x1 - x1 * x1 + x1 * x2,
        l1 * x1 * y1 - x1 * y1 + x2 * y1, 
        l1 * x1 - x1 + x2,
        l1 * x1 * y1 - x1 * y1 + x1 * y2,
        l1 * y1 * y1 - y1 * y1 + y1 * y2, 
        l1 * y1 - y1 + y2, 
        x1,
        y1, 
        1;
    A.template block<1,9>(row,0) << 
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

Eigen::Matrix<Eigen::Matrix<double, 1, 2>, 9, 9> MergeAB(Eigen::Matrix<double, 9, 9> A,
                                                         Eigen::Matrix<double, 9, 9> B) {
    Eigen::Matrix<Eigen::Matrix<double, 1, 2>, 9, 9> ret;
    for (int i = 0; i < 9 * 9; ++i) {
        ret.data()[i] << B.data()[i], A.data()[i];
    }
    return ret;
}

double SolveForK(Eigen::Matrix<double, 9, 9> A, Eigen::Matrix<double, 9, 9> B) {
    static const Det99Computer comp;

    auto P = MergeAB(A, B);

    // for (int i = 0; i < 9; ++i) {
    //     for (int j = 0; j < 9; ++j) {
    //         P(i, j) << (i == j), 0.;
    //         // std::cout << P(i,j) << std::endl;
    //     }
    // }

    // for (int i = 0; i < 9; ++i) {
    //     for (int j = 0; j < 9; ++j) {
    //         // P(i,j) << (i==j), 0.;
    //         std::cout << P(i, j) << std::endl;
    //     }
    // }

    auto det_P = comp.Compute(P);

    // std::cout << det_P << std::endl;

    int deg_P = 6;
    while (deg_P >= 0 && fabs(det_P[deg_P]) < 1e-20) {
        --deg_P;
    }

    if (deg_P <= 0) {
        return 0.;
    }

    det_P /= det_P[deg_P];

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(deg_P, deg_P);

    for (int j = 0; j < deg_P; ++j) {
        if (j + 1 < deg_P) C(j + 1, j) = 1;
        C(j, deg_P - 1) = -det_P[j];
    }

    // std::cout << Eigen::JacobiSVD<decltype(C)>(C).singularValues().transpose() << std::endl;

    return Eigen::JacobiSVD<decltype(C)>(C).singularValues().tail(1)[0];
}

template <class T>
void NormailzePointSet(T& points, Eigen::Matrix3d& reverse) {
    double scale = 0;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();

    for (int i = 0; i < points.size(); ++i) {
        center += points[i].template block<2, 1>(0, 0);
    }
    center /= points.size();

    for (int i = 0; i < points.size(); ++i) {
        scale += (points[i].template block<2, 1>(0, 0) - center).norm();
    }
    scale /= points.size();

    scale = sqrt(2) / scale;

    for (int i = 0; i < points.size(); ++i) {
        points[i].template block<2, 1>(0, 0) = (Eigen::Vector2d{points[i].data()} - center) * scale;
    }

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

    // std::cout << A << std::endl << std::endl << B << std::endl << std::endl;

    double k = SolveForK(A, B);

    k = 0;

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
        return ::ComputeError(datum.first, datum.second, model.first);
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