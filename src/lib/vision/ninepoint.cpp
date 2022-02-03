#include <Eigen/Eigen>
#include <rtl/RTL.hpp>

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include <iostream>
#include <iomanip>

namespace {

template <class T>
void SetRow(int row, T& A, T& B, Eigen::Vector2d point1, Eigen::Vector2d point2, double t1,
            double t2) {
    point1 /= sqrt(point1.squaredNorm() + 1);
    point2 /= sqrt(point2.squaredNorm() + 1);
    double l1 = t2 - t1;
    double l2 = l1 * (t2 + t1 + 1.) / 2.;
    double u_x = point2[0] - point1[0], u_y = point2[1] - point1[1], u_z = 0;
    double x = point1[0], y = point1[1], z = 1;

    // clang-format off
    B.template block<1,9>(row,0) << 
        -l1*x*x,
        -2*l1*x*y,
        -2*l1*x*z,
        -l1*y*y,
        -2*l1*y*z,
        -l1*z*z,
        -u_y*z + u_z*y,
        u_x*z - u_z*x,
        -u_x*y + u_y*x;
    A.template block<1,9>(row,0) << 
        -l2*x*x, -2*l2*x*y, -2*l2*x*z, -l2*y*y, -2*l2*y*z, -l2*z*z, 0, 0, 0;
    // clang-format on
    // A.template block<1,9>(row,0) *= 100;
    // B.template block<1,9>(row,0) *= 100;
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
    return 0;
    if (fabs(B.determinant()) < 1e-20) return 0;

    // Eigen::MatrixXd Bxd = B;
    // auto svd = Bxd.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 9> B_inv = B.inverse();
    // svd.matrixV() * svd.singularValues().asDiagonal().inverse() * svd.matrixU().transpose();

    int max_real_i = -1;
    double best_k = 100000;
    Eigen::Matrix<std::complex<double>, 9, 1> eigens = (A * B_inv).eigenvalues();
    for (int i = 0; i < 9; ++i) {
        double k = fabs(-1. / eigens(i, 0).real());
        if (fabs(eigens(i, 0).imag()) < 1e-20 && k > 1e-9 && k < best_k) {
            max_real_i = i;
            best_k = k;
        }
    }

    if (max_real_i < 0) {
        return 0;
    }

    double solution = -1 / eigens(max_real_i, 0).real();

    // std::cout << eigens.transpose() << std::endl;
    // std::cout << solution << std::endl;

    if (!std::isfinite(solution)) {
        return 0.;
    }

    return solution;
}

template <class T>
void NormailzePointSet(T& points, Eigen::Matrix3d& reverse) {
    reverse = Eigen::Matrix3d::Identity();
    return;

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
double ComputeError(const T& point1, const T& point2, std::pair<Eigen::Matrix3d, double> model) {
    return 0;
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

    double k = SolveForK(A, B);

    if (out_k) {
        out_k[0] = k;
    }

    Eigen::Matrix<double, 3, 3> Q;
    Eigen::Vector3d V0;
    {
        Eigen::Matrix<double, 9, 9> D = B + A * k;
        Eigen::JacobiSVD<decltype(D)> solver(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 9, 1> m(solver.matrixV().col(8).real().data());
        V0 = m.block<3, 1>(6, 0);
        Q << m(0, 0), m(1, 0), m(2, 0), m(1, 0), m(3, 0), m(4, 0), m(2, 0), m(4, 0), m(5, 0);
        Q /= V0.norm();

        // std::cout << D * m << std::endl;;
        // std::cout << "det D = " << D.determinant() << std::endl;
        // std::cout << m(6, 0) << " " << m(7, 0) << " " << m(8, 0) << std::endl;
        // std::cout << D << "\n--------------------\n" << std::endl;
    }
    {
        Eigen::JacobiSVD<decltype(Q)> solver(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 3, 1> lambdas = solver.singularValues(), sigmas;
        // clang-format off
        sigmas << 
            (2.*lambdas(0,0) + lambdas(1,0) - lambdas(2,0)) / 3.,
            (2.*lambdas(1,0) + lambdas(0,0) + lambdas(2,0)) / 3.,
            (2.*lambdas(2,0) + lambdas(1,0) - lambdas(0,0)) / 3.;

        double 
            lambda = sigmas(0,0) - sigmas(2,0),
            theta = acos(-sigmas(1,0) / lambda);

        // std::cout << std::fixed << std::setprecision(15) << lambda << " " << theta << std::endl;

        Eigen::Matrix3d 
            V = solver.matrixV() * Eigen::AngleAxisd(theta / 2. - M_PI / 2., Eigen::Vector3d{0, 1, 0}).toRotationMatrix().transpose(),
            U = V * Eigen::AngleAxisd(theta, Eigen::Vector3d{0, 1, 0}).toRotationMatrix(),
            RZpi2p = Eigen::AngleAxisd(M_PI / 2., Eigen::Vector3d{0, 0, 1}).toRotationMatrix(),
            RZpi2n = Eigen::AngleAxisd(-M_PI / 2., Eigen::Vector3d{0, 0, 1}).toRotationMatrix(),
            sigma_1 = Eigen::Vector3d{1,1,0}.asDiagonal(),
            sigma_lambda = sigma_1 * lambda;

        Eigen::Matrix3d
            omega1hat = U * RZpi2p * sigma_lambda * U.transpose(),
            omega2hat = V * RZpi2p * sigma_lambda * V.transpose(),
            omega3hat = U * RZpi2n * sigma_lambda * U.transpose(),
            omega4hat = V * RZpi2n * sigma_lambda * V.transpose();

        Eigen::Matrix3d
            v1hat = V * RZpi2p * sigma_1 * V.transpose(),
            v2hat = U * RZpi2p * sigma_1 * U.transpose(),
            v3hat = V * RZpi2n * sigma_1 * V.transpose(),
            v4hat = U * RZpi2n * sigma_1 * U.transpose();


        omega1hat = (omega1hat - omega1hat.transpose()) / 2.;
        omega2hat = (omega2hat - omega2hat.transpose()) / 2.;
        omega3hat = (omega3hat - omega3hat.transpose()) / 2.;
        omega4hat = (omega4hat - omega4hat.transpose()) / 2.;

        v1hat = (v1hat - v1hat.transpose()) / 2.;
        v2hat = (v2hat - v2hat.transpose()) / 2.;
        v3hat = (v3hat - v3hat.transpose()) / 2.;
        v4hat = (v4hat - v4hat.transpose()) / 2.;

        Eigen::Vector3d omegas[4], vs[4];
        omegas[0] << - omega1hat(1,2), omega1hat(0,2), - omega1hat(0,1);
        omegas[1] << - omega2hat(1,2), omega2hat(0,2), - omega2hat(0,1);
        omegas[2] << - omega3hat(1,2), omega3hat(0,2), - omega3hat(0,1);
        omegas[3] << - omega4hat(1,2), omega4hat(0,2), - omega4hat(0,1);

        vs[0] << - v1hat(1,2), v1hat(0,2), - v1hat(0,1);
        vs[1] << - v2hat(1,2), v2hat(0,2), - v2hat(0,1);
        vs[2] << - v3hat(1,2), v3hat(0,2), - v3hat(0,1);
        vs[3] << - v4hat(1,2), v4hat(0,2), - v4hat(0,1);

        int best_i = 0;
        for (int i = 0; i < 4; ++i) {
            if (fabs(vs[i].normalized().dot(V0)) > vs[best_i].normalized().dot(V0)) {
                best_i = i;
            }
        }
        // std::cout << omegas[best_i].normalized().dot(V0) << std::endl;


        // std::cout << (omega1hat - omega1hat.transpose()) / 2. << std::endl << std::endl << (omega2hat - omega2hat.transpose()) / 2. << std::endl;
        // std::cout << (omega1hat - omega1hat.transpose()) / 2. << std::endl << std::endl << (omega2hat - omega2hat.transpose()) / 2. << std::endl;
        std::cout << omegas[0].norm() << " " << omegas[1].norm() << " " << omegas[2].norm() << " " << omegas[3].norm() << std::endl;
        // std::cout << omegas[0].transpose() << "\n" << omegas[1].transpose() << "\n" << omegas[2].transpose() << "\n" << omegas[3].transpose() << "\n" << std::endl;
        // clang-format on
        // std::cout << "L " << lambda << " " << theta << std::endl;
        // std::cout << "L " << lambdas.transpose() << "\n--------------------------\n";
        // std::cout << sigmas.transpose() << "\n--------------------------\n";
        // std::cout << "\n--------------------------\n";
    }

    return {};  // T2.transpose() * E * T1;
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

    Eigen::MatrixXd D = B;// + A * k;

    Eigen::Matrix<double, 3, 3> Q;
    Eigen::Vector3d V0;
    {
        Eigen::Matrix<double, 9, 9> D = B + A * k;
        Eigen::JacobiSVD<decltype(D)> solver(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 9, 1> m(solver.matrixV().col(8).real().data());
        V0 = m.block<3, 1>(6, 0);
        Q << m(0, 0), m(1, 0), m(2, 0), m(1, 0), m(3, 0), m(4, 0), m(2, 0), m(4, 0), m(5, 0);
        Q /= V0.norm();

        // std::cout << D * m << std::endl;;
        // std::cout << "det D = " << D.determinant() << std::endl;
        // std::cout << m(6, 0) << " " << m(7, 0) << " " << m(8, 0) << std::endl;
        // std::cout << D << "\n--------------------\n" << std::endl;
    }
    {
        Eigen::JacobiSVD<decltype(Q)> solver(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 3, 1> lambdas = solver.singularValues(), sigmas;
        // clang-format off
        sigmas << 
            (2.*lambdas(0,0) + lambdas(1,0) - lambdas(2,0)) / 3.,
            (2.*lambdas(1,0) + lambdas(0,0) + lambdas(2,0)) / 3.,
            (2.*lambdas(2,0) + lambdas(1,0) - lambdas(0,0)) / 3.;

        double 
            lambda = sigmas(0,0) - sigmas(2,0),
            theta = acos(-sigmas(1,0) / lambda);

        // std::cout << std::fixed << std::setprecision(15) << lambda << " " << theta << std::endl;

        Eigen::Matrix3d 
            V = solver.matrixV() * Eigen::AngleAxisd(theta / 2. - M_PI / 2., Eigen::Vector3d{0, 1, 0}).toRotationMatrix().transpose(),
            U = V * Eigen::AngleAxisd(theta, Eigen::Vector3d{0, 1, 0}).toRotationMatrix(),
            RZpi2p = Eigen::AngleAxisd(M_PI / 2., Eigen::Vector3d{0, 0, 1}).toRotationMatrix(),
            RZpi2n = Eigen::AngleAxisd(-M_PI / 2., Eigen::Vector3d{0, 0, 1}).toRotationMatrix(),
            sigma_1 = Eigen::Vector3d{1,1,0}.asDiagonal(),
            sigma_lambda = sigma_1 * lambda;

        Eigen::Matrix3d
            omega1hat = U * RZpi2p * sigma_lambda * U.transpose(),
            omega2hat = V * RZpi2p * sigma_lambda * V.transpose(),
            omega3hat = U * RZpi2n * sigma_lambda * U.transpose(),
            omega4hat = V * RZpi2n * sigma_lambda * V.transpose();

        Eigen::Matrix3d
            v1hat = V * RZpi2p * sigma_1 * V.transpose(),
            v2hat = U * RZpi2p * sigma_1 * U.transpose(),
            v3hat = V * RZpi2n * sigma_1 * V.transpose(),
            v4hat = U * RZpi2n * sigma_1 * U.transpose();


        omega1hat = (omega1hat - omega1hat.transpose()) / 2.;
        omega2hat = (omega2hat - omega2hat.transpose()) / 2.;
        omega3hat = (omega3hat - omega3hat.transpose()) / 2.;
        omega4hat = (omega4hat - omega4hat.transpose()) / 2.;

        v1hat = (v1hat - v1hat.transpose()) / 2.;
        v2hat = (v2hat - v2hat.transpose()) / 2.;
        v3hat = (v3hat - v3hat.transpose()) / 2.;
        v4hat = (v4hat - v4hat.transpose()) / 2.;

        Eigen::Vector3d omegas[4], vs[4];
        omegas[0] << - omega1hat(1,2), omega1hat(0,2), - omega1hat(0,1);
        omegas[1] << - omega2hat(1,2), omega2hat(0,2), - omega2hat(0,1);
        omegas[2] << - omega3hat(1,2), omega3hat(0,2), - omega3hat(0,1);
        omegas[3] << - omega4hat(1,2), omega4hat(0,2), - omega4hat(0,1);

        vs[0] << - v1hat(1,2), v1hat(0,2), - v1hat(0,1);
        vs[1] << - v2hat(1,2), v2hat(0,2), - v2hat(0,1);
        vs[2] << - v3hat(1,2), v3hat(0,2), - v3hat(0,1);
        vs[3] << - v4hat(1,2), v4hat(0,2), - v4hat(0,1);

        int best_i = 0;
        for (int i = 0; i < 4; ++i) {
            if (fabs(vs[i].normalized().dot(V0)) > vs[best_i].normalized().dot(V0)) {
                best_i = i;
            }
        }
        // std::cout << omegas[best_i].normalized().dot(V0) << std::endl;


        // std::cout << (omega1hat - omega1hat.transpose()) / 2. << std::endl << std::endl << (omega2hat - omega2hat.transpose()) / 2. << std::endl;
        // std::cout << (omega1hat - omega1hat.transpose()) / 2. << std::endl << std::endl << (omega2hat - omega2hat.transpose()) / 2. << std::endl;
        std::cout << omegas[0].norm() << " " << omegas[1].norm() << " " << omegas[2].norm() << " " << omegas[3].norm() << std::endl;
    }

    return {};//T2.transpose() * E * T1;
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

    if (k_out) {
        *k_out = model.second;
    }

    // return model.first;

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

    return model.first;
}