#pragma once
#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>

#include <ceres/jet.h>

template <class T>
class GenericQuaternionGroup;

template <typename T>
inline double Real(const T& x) {
    return x;
}

template <typename T, int N>
inline T Real(const ceres::Jet<T, N>& x) {
    return x.a;
}

template <class T>
class GenericQuaternion {
    static constexpr double eps = 1e-15;

    friend class GenericQuaternionGroup<T>;

   public:
    template <class S>
    explicit GenericQuaternion(S a, S b, S c, S d) : GenericQuaternion{T(a), T(b), T(c), T(d)} {}

    template <class S>
    explicit GenericQuaternion(S rx, S ry, S rz) : GenericQuaternion{T(rx), T(ry), T(rz)} {}

    GenericQuaternion() { std::fill(data_, data_ + 4, T{}); }

    explicit GenericQuaternion(T a, T b, T c, T d) : data_{a, b, c, d} {}

    explicit GenericQuaternion(T rx, T ry, T rz) {
        auto norm = sqrt(rx * rx + ry * ry + rz * rz);
        auto a = cos(norm / 2.);
        auto k = SinxInvx(norm / 2.) / 2.;
        auto b = rx * k;
        auto c = ry * k;
        auto d = rz * k;
        if (Real(norm) < eps) {
            a = T{1.};
            b = rx;
            c = ry;
            d = rz;
        }
        data_[0] = a;
        data_[1] = b;
        data_[2] = c;
        data_[3] = d;
    }

    T Norm() const {
        return sqrt(data_[0] * data_[0] + data_[1] * data_[1] + data_[2] * data_[2] +
                    data_[3] * data_[3]);
    }

    void ToRotVec(T& rx, T& ry, T& rz) const {
        auto cos = data_[0];
        auto sin_norm = sqrt(data_[1] * data_[1] + data_[2] * data_[2] + data_[3] * data_[3]);
        auto angle = 2. * atan2(sin_norm, cos);
        if (Real(sin_norm) < eps) {
            rx = ry = rz = {};
            return;
        }
        rx = data_[1] / sin_norm * angle;
        ry = data_[2] / sin_norm * angle;
        rz = data_[3] / sin_norm * angle;
    }

    GenericQuaternion<double> Bias(double x, double y, double z) {
        T a{data_[0]}, b{data_[1]}, c{data_[2]}, d{data_[3]};
        return GenericQuaternion<double>{
            a.a + a.v[0] * x + a.v[1] * y + a.v[2] * z,
            b.a + b.v[0] * x + b.v[1] * y + b.v[2] * z,
            c.a + c.v[0] * x + c.v[1] * y + c.v[2] * z,
            d.a + d.v[0] * x + d.v[1] * y + d.v[2] * z,
        };
    }

   private:
    T data_[4];

    T SinxInvx(T x) const {
        if (fabs(Real(x)) < eps) {
            return -x * x / 6.;
        }
        return sin(x) / x;
    }
    
    template <class Q>
    friend std::ostream& operator<<(std::ostream& s, const GenericQuaternion<Q>& q);
};

using Quaternion = GenericQuaternion<double>;

template <class T>
inline std::ostream& operator<<(std::ostream& s, const GenericQuaternion<T>& q) {
    // T x, y, z;
    // q.ToRotVec(x, y, z);
    // T norm = sqrt(x * x + y * y + z * z);
    // s << "[Rotation " << norm * 180. / M_PI << "; " << x << " " << y << " " << z << "]";
    s << "[" << q.data_[0] << " " << q.data_[1] << " " << q.data_[2] << " " << q.data_[3] << "]";
    return s;
}

template <class T>
struct GenericQuaternionGroup {
    typedef GenericQuaternion<T> value_type;

    GenericQuaternion<T> unit() const { return GenericQuaternion<T>{1, 0, 0, 0}; }

    GenericQuaternion<T> add(GenericQuaternion<T> a, GenericQuaternion<T> b) const {
        // (a0 + b0*i + c0*j + d0*k) * (a1 + b1*i + c1*j + d1*k) =
        // (a0*a1 + a0*b1*i + a0*c1*j + a0*d1*k) (b0*a1*i + b0*b1*-1 + b0*c1*k +
        // b0*d1*-j) (c0*a1*j + c0*b1*-k + c0*c1*-1 + c0*d1*i)(d0*a1*k + d0*b1*j
        // + d0*c1*-i + d0*d1*-1) = (a0*a1 + b0*b1*-1 + c0*c1*-1  + d0*d1*-1) +
        // (a0*b1 + b0*a1 + c0*d1 - d0*c1)*i + (a0*c1 - b0*d1 + c0*a1 + d0*b1)*j
        // + (a0*d1 + b0*c1 - c0*b1 + d0*a1)*k
        T a0{a.data_[0]}, b0{a.data_[1]}, c0{a.data_[2]}, d0{a.data_[3]};
        T a1{b.data_[0]}, b1{b.data_[1]}, c1{b.data_[2]}, d1{b.data_[3]};
        return GenericQuaternion<T>{
            a0 * a1 - b0 * b1 - c0 * c1 - d0 * d1, a0 * b1 + b0 * a1 + c0 * d1 - d0 * c1,
            a0 * c1 - b0 * d1 + c0 * a1 + d0 * b1, a0 * d1 + b0 * c1 - c0 * b1 + d0 * a1};
    }

    GenericQuaternion<T> mult(GenericQuaternion<T> a, double k) const {
        T x, y, z;
        a.ToRotVec(x, y, z);
        x *= k;
        y *= k;
        z *= k;
        return GenericQuaternion<T>(x, y, z);
    }

    GenericQuaternion<T> inv(GenericQuaternion<T> q) const {
        return GenericQuaternion{q.data_[0], -q.data_[1], -q.data_[2], -q.data_[3]};
    }
};

using QuaternionGroup = GenericQuaternionGroup<double>;

template <class T>
struct DefaultGroup {
    typedef T value_type;
    T unit() const { return {}; }

    T add(const T& a, const T& b) const { return a + b; }

    T mult(const T& a, double k) const { return a * k; }

    T inv(const T& a) const { return -a; }
};
