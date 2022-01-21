#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <numeric>

#include <math/simple_math.hpp>

SCENARIO("Approximation of sum with bias") {
    using GQ = GenericQuaternion<ceres::Jet<double, 3>>;
    using GQG = GenericQuaternionGroup<ceres::Jet<double, 3>>;
    using Q = Quaternion;
    using QG = QuaternionGroup;

    const size_t size = 33;
    const double bx{5e-3}, by{1e-3}, bz{-4e-3};
    const double rx{1e-1}, ry{-2e-2}, rz{1e-3};

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

    double g_x, g_y, g_z;
    double x, y, z;

    g_integrated.Bias(bx, by, bz).ToRotVec(g_x, g_y, g_z);
    integrated.ToRotVec(x, y, z);

    auto norm = sqrt(bx * bx + by * by + bz * bz);
    REQUIRE(g_x == (Approx(x).margin(fabs(size * norm / 20))));
    REQUIRE(g_y == (Approx(y).margin(fabs(size * norm / 20))));
    REQUIRE(g_z == (Approx(z).margin(fabs(size * norm / 20))));

    // std::cout << g_x << " " << g_y << " " << g_z << std::endl;
    // std::cout << x << " " << y << " " << z << std::endl;
    // std::cout << size * bx << " " << size * by << " " << size * bz << std::endl;
}
