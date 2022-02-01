#include <vision/ninepoint.hpp>

#include <vector>
#include <chrono>

#include <Eigen/Eigen>

#include <iostream>

#include <algorithm>
#include <array>
#include <tuple>

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
    srand((unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    double k;
    std::vector<unsigned char> mask;
    std::vector<Eigen::Vector3d> points1, points2;

    Eigen::AngleAxis<double> rot(.4, Eigen::Vector3d{1, 0, 0});
    for (int i = 0; i < 90; ++i) {
        Eigen::Vector3d p1 = Eigen::Vector3d::Random();
        Eigen::Vector3d shift{0, 1, .0};
        Eigen::Vector3d p2 = rot * (p1 + shift) + Eigen::Vector3d::Random() * .01;
        p1 /= p1[2];
        p2 /= p2[2];
        // p2[0] *= .8;
        // p2[1] *= .8;
        // p2[0] += .01;
        // p2[1] += .01;
        p1[2] = 0;
        p2[2] = 1;
        // p1[2] = p1[1];
        // p2[2] = 1 + p2[1];

        points1.push_back(p1);
        points2.push_back(p2);
    }

    mask.resize(points1.size());
    std::fill(mask.begin(), mask.end(), 1);

    auto E = FindEssentialMat(points1, points2, mask, 1e-7, 100, &k);

    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    DecomposeEssential(E, R1, R2, t);
    std::cout << "R1:\n" << R1 << "\n\nR2:\n" << R2 << "\n\nt:\n" << t << std::endl;
    Eigen::AngleAxis<double> r1, r2;
    r1.fromRotationMatrix(R1);
    r2.fromRotationMatrix(R2);
    std::cout << "R1: " << r1.angle() << "  R2: " << r2.angle() << std::endl;
    std::cout << "k: " << k << std::endl;

    return 0;
}