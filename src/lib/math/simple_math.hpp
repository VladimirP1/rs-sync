#pragma once
#include <algorithm>
#include <cmath>

class QuaternionGroup;

class Quaternion {
    static constexpr double eps = 1e-15;

    friend class QuaternionGroup;

   public:
    Quaternion() { std::fill(data_, data_ + 4, 0); }

    explicit Quaternion(double a, double b, double c, double d)
        : data_{a, b, c, d} {}

    explicit Quaternion(double rx, double ry, double rz) {
        auto norm = std::sqrt(rx * rx + ry * ry + rz * rz);
        auto a = cos(norm / 2.);
        auto k = SinxInvx(norm / 2) / 2.;
        auto b = rx * k;
        auto c = ry * k;
        auto d = rz * k;
        data_[0] = a;
        data_[1] = b;
        data_[2] = c;
        data_[3] = d;
    }

    double Norm() {
        return std::sqrt(data_[0] * data_[0] + data_[1] * data_[1] +
                         data_[2] * data_[2] + data_[3] * data_[3]);
    }

    void ToRotVec(double& rx, double& ry, double& rz) {
        auto cos = data_[0];
        auto sin_norm = std::sqrt(data_[1] * data_[1] + data_[2] * data_[2] +
                                  data_[3] * data_[3]);
        auto angle = 2 * atan2(sin_norm, cos);
        if (sin_norm < eps) {
            rx = ry = rz = 0.;
            return;
        }
        rx = data_[1] / sin_norm * angle;
        ry = data_[2] / sin_norm * angle;
        rz = data_[3] / sin_norm * angle;
    }

   private:
    double data_[4];

    double SinxInvx(double x) {
        if (std::fabs(x) < eps) {
            return 1.;
        }
        return std::sin(x) / x;
    }
};

struct QuaternionGroup {
    typedef Quaternion value_type;

    Quaternion unit() const { return Quaternion{1, 0, 0, 0}; }

    Quaternion add(Quaternion a, Quaternion b) const {
        // (a0 + b0*i + c0*j + d0*k) * (a1 + b1*i + c1*j + d1*k) =
        // (a0*a1 + a0*b1*i + a0*c1*j + a0*d1*k) (b0*a1*i + b0*b1*-1 + b0*c1*k +
        // b0*d1*-j) (c0*a1*j + c0*b1*-k + c0*c1*-1 + c0*d1*i)(d0*a1*k + d0*b1*j
        // + d0*c1*-i + d0*d1*-1) = (a0*a1 + b0*b1*-1 + c0*c1*-1  + d0*d1*-1) +
        // (a0*b1 + b0*a1 + c0*d1 - d0*c1)*i + (a0*c1 - b0*d1 + c0*a1 + d0*b1)*j
        // + (a0*d1 + b0*c1 - c0*b1 + d0*a1)*k
        double a0{a.data_[0]}, b0{a.data_[1]}, c0{a.data_[2]}, d0{a.data_[3]};
        double a1{b.data_[0]}, b1{b.data_[1]}, c1{b.data_[2]}, d1{b.data_[3]};
        return Quaternion{a0 * a1 - b0 * b1 - c0 * c1 - d0 * d1,
                          a0 * b1 + b0 * a1 + c0 * d1 - d0 * c1,
                          a0 * c1 - b0 * d1 + c0 * a1 + d0 * b1,
                          a0 * d1 + b0 * c1 - c0 * b1 + d0 * a1};
    }

    Quaternion mult(Quaternion a, double k) const {
        double x, y, z;
        a.ToRotVec(x, y, z);
        x *= k;
        y *= k;
        z *= k;
        return Quaternion(x, y, z);
    }

    Quaternion inv(Quaternion q) const {
        return Quaternion{q.data_[0], -q.data_[1], -q.data_[2], -q.data_[3]};
    }
};

template <class T>
struct DefaultGroup {
    typedef T value_type;
    T unit() const { return {}; }

    T add(const T& a, const T& b) const { return a + b; }

    T mult(const T& a, double k) const { return a * k; }

    T inv(const T& a) const { return -a; }
};