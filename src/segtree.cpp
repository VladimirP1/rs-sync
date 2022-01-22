
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <math/quaternion.hpp>

int main(int argc, char** argv) {
    // auto q0 = Quaternion<ceres::Jet<double, 3>>::FromRotationVector({0, 0}, {0, 1}, {0, 2});

    // std::cout << q0 << std::endl;
    // std::cout << (q0 * q0) << std::endl;

    // ceres::Jet<double, 3> x, y, z;
    // q0.ToRotationVector(x, y, z);
    // std::cout << x << " " << y << " " << z << std::endl;

    // QuaternionGroup<decltype(q0)> g;
    // g.unit();
    // q0 = g.add(q0, q0);
    // q0 = g.inv(q0);
    // q0 = g.mult(q0, 1);

    auto q0 = Quaternion<ceres::Jet<double, 3>>::FromRotationVector({{0, 0}, {0, 1}, {0, 2}});
    auto q1 = Quaternion<ceres::Jet<double, 3>>::FromRotationVector(
        {Jet<double, 3>{.01}, Jet<double, 3>{0}, Jet<double, 3>{0}});

    auto b = GetBiasForOffset(q1 * q0);
    std::cout << b << std::endl;

    std::cout << Bias(q1 * q0, b) << std::endl;

    // double x, y, z;
    // Bias(q1*q0,bx,by,bz).ToRotationVector(x, y, z);
    // std::cout << x << " " << y << " " << z << std::endl;

    /*
    using GQ = GenericQuaternion<ceres::Jet<double, 3>>;
    using GQG = GenericQuaternionGroup<ceres::Jet<double, 3>>;
    using Q = Quaternion;
    using QG = QuaternionGroup;

    const size_t size = 33;
    const double bx{5e-3}, by{1e-3}, bz{-4e-3};
    const double rx{1e-1}, ry{-2e-2}, rz{0};

    GQG g_g;
    std::vector<GQ> g_quat(size, GQ{{rx, 0}, {ry, 1}, {rz, 2}});
    auto g_integrated = g_g.unit();
    for (auto& q : g_quat) {
        g_integrated = g_g.add(q, g_integrated);
    }

    QG g;
    std::vector<Q> quat(size, Q{rx + bx, ry + by, rz + bz});
    auto integrated = g.unit();
    for (auto& q : quat) {
        integrated = g.add(q, integrated);
    }

    double x, y, z;

    g_integrated.Bias(bx, by, bz).ToRotVec(x, y, z);
    std::cout << x << " " << y << " " << z << std::endl;

    integrated.ToRotVec(x, y, z);
    std::cout << x << " " << y << " " << z << std::endl;

    std::cout << size * bx << " " << size * by << " " << size * bz << std::endl;

    std::cout << fabs(size * bx / 50) << std::endl;*/
    // QuaternionJet j(ceres::Jet<double,4>(1,
    // 0),ceres::Jet<double,4>(1),ceres::Jet<double,4>(1)); QuaternionGroupJet g; auto q =
    // g.unit(); std::cout << q << std::endl; ceres::Jet<double, 4> x,y,z; q.ToRotVec(x,y,z);

    return 0;
}