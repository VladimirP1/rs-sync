
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <io/bb_csv.hpp>

#include <math/simple_math.hpp>
#include <ds/prefix_sums.hpp>
#include <ds/segment_tree.hpp>
#include <ds/range_interpolated.hpp>

int main(int argc, char** argv) {
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

    std::cout << fabs(size * bx / 50) << std::endl;
    // QuaternionJet j(ceres::Jet<double,4>(1, 0),ceres::Jet<double,4>(1),ceres::Jet<double,4>(1));
    // QuaternionGroupJet g;
    // auto q = g.unit();
    // std::cout << q << std::endl;
    // ceres::Jet<double, 4> x,y,z;
    // q.ToRotVec(x,y,z);

    return 0;
}