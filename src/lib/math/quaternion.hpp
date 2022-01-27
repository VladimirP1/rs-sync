#pragma once
#include <ostream>

#include <cmath>
#include <ceres/jet.h>
#define EIGEN_QUATERNION_PLUGIN <math/__quaternion_plugin.h>
#include <Eigen/Eigen>
#undef EIGEN_QUATERNION_PLUGIN

using Eigen::Quaternion;
using Eigen::Matrix;
using ceres::Jet;

template <class T>
std::ostream& operator<<(std::ostream& s, const Eigen::Quaternion<T>& q) {
    s << q.w() << " " << q.x() << " " << q.y() << " " << q.z();
    return s;
}

template <class T>
struct QuaternionGroup {
    typedef T value_type;

    T unit() const { return value_type::Identity(); }

    T add(T a, T b) const {
        return a * b;
    }

    T mult(T a, double k) const {
        auto v = a.ToRotationVector() * k;
        return T::FromRotationVector(v);
    }

    T inv(T q) const {
        return q.inverse();
    }
};

inline Quaternion<double> Bias(const Quaternion<Jet<double, 3>>& q, Eigen::Matrix<double, 3, 1> bv) {
    Jet<double, 3> a{q.w()}, b{q.x()}, c{q.y()}, d{q.z()};
    return Quaternion<double>{
        a.a + a.v[0] * bv.x() + a.v[1] * bv.y() + a.v[2] * bv.z(),
        b.a + b.v[0] * bv.x() + b.v[1] * bv.y() + b.v[2] * bv.z(),
        c.a + c.v[0] * bv.x() + c.v[1] * bv.y() + c.v[2] * bv.z(),
        d.a + d.v[0] * bv.x() + d.v[1] * bv.y() + d.v[2] * bv.z(),
    };
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