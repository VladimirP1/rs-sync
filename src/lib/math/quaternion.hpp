#pragma once
#include <ostream>

#include <cmath>
#include <ceres/jet.h>
#define EIGEN_QUATERNION_PLUGIN <math/__quaternion_plugin.h>
#include <Eigen/Eigen>
#undef EIGEN_QUATERNION_PLUGIN

using ceres::Jet;
using Eigen::Matrix;
using Eigen::Quaternion;

template <class T>
std::ostream& operator<<(std::ostream& s, const Eigen::Quaternion<T>& q) {
    s << q.w() << " " << q.x() << " " << q.y() << " " << q.z();
    return s;
}

template <class T>
struct QuaternionGroup {
    typedef T value_type;

    T unit() const { return value_type::Identity(); }

    T add(T a, T b) const { return a * b; }

    T mult(T a, double k) const {
        auto v = a.ToRotationVector() * k;
        return T::FromRotationVector(v);
    }

    T inv(T q) const { return q.inverse(); }
};

inline Eigen::Matrix3d BiasJacobian(const Quaternion<Jet<double, 3>>& q) {

    auto rv = q.ToRotationVector();
    Eigen::Matrix3d jac;
    // clang-format off
    jac << 
        rv.x().v[0], rv.x().v[1], rv.x().v[2],
        rv.y().v[0], rv.y().v[1], rv.y().v[2], 
        rv.z().v[0], rv.z().v[1], rv.z().v[2];
    // clang-format on
    return jac;
}

inline Quaternion<double> Bias(const Quaternion<Jet<double, 3>>& q,
                               Eigen::Matrix<double, 3, 1> bv) {
    Jet<double, 3> a{q.w()}, b{q.x()}, c{q.y()}, d{q.z()};
    return Quaternion<double>{a.a + a.v.dot(bv), b.a + b.v.dot(bv), c.a + c.v.dot(bv),
                              d.a + d.v.dot(bv)};
}

inline Eigen::Matrix<double, 3, 1> GetBiasForOffset(const Quaternion<Jet<double, 3>>& error) {
    auto e = error.ToRotationVector();
    Eigen::Matrix<double, 3, 1> ea{{e.x().a}, {e.y().a}, {e.z().a}};
    Eigen::Matrix3d jac;
    // clang-format off
    jac << 
        e.x().v[0], e.x().v[1], e.x().v[2],
        e.y().v[0], e.y().v[1], e.y().v[2], 
        e.z().v[0], e.z().v[1], e.z().v[2];
    // clang-format on

    return jac.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-ea);
    // return jac.colPivHouseholderQr().solve(-ea);
    // return jac.inverse() * (-ea);
}