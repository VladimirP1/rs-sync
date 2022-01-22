#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <numeric>

#include <math/quaternion.hpp>

SCENARIO("Bias smoke test") {
    auto q0 = Quaternion<ceres::Jet<double, 3>>::FromRotationVector({{0, 0}, {0, 1}, {0, 2}});
    auto q1 = Quaternion<ceres::Jet<double, 3>>::FromRotationVector(
        {Jet<double, 3>{.01}, Jet<double, 3>{0}, Jet<double, 3>{0}});

    auto b = GetBiasForOffset(q1 * q0);

    REQUIRE(Bias(q1 * q0, b).ToRotationVector().norm() < .0001);
}

SCENARIO("Approximation of sum with bias") {
    using GQ = Quaternion<Jet<double, 3>>;
    using GQG = QuaternionGroup<GQ>;
    using Q = Quaternion<double>;
    using QG = QuaternionGroup<Q>;

    const size_t size = 33;
    const Matrix<double, 3, 1> b{5e-3, 1e-3, -4e-3};
    const Matrix<double, 3, 1> r{1e-1, -2e-2, 1e-3};

    GQG g_g;
    std::vector<GQ> g_quat(size, GQ::FromRotationVector({{r.x(), 0}, {r.y(), 1}, {r.z(), 2}}));
    auto g_integrated = g_g.unit();
    for (auto& q : g_quat) {
        g_integrated = g_g.add(q, g_integrated);
    }

    QG g;
    std::vector<Q> quat(size, Q::FromRotationVector(r + b));
    auto integrated = g.unit();
    for (auto& q : quat) {
        integrated = g.add(q, integrated);
    }

    auto gv = Bias(g_integrated, b).ToRotationVector();
    auto v = integrated.ToRotationVector();

    auto norm = b.norm();
    REQUIRE(gv.x() == (Approx(v.x()).margin(fabs(size * norm / 20))));
    REQUIRE(gv.y() == (Approx(v.y()).margin(fabs(size * norm / 20))));
    REQUIRE(gv.z() == (Approx(v.z()).margin(fabs(size * norm / 20))));
}
