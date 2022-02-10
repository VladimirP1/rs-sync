#pragma once

#include <ds/segment_tree.hpp>

#include <Eigen/Eigen>
#include <unsupported/Eigen/AutoDiff>

/* The following three functions are adopted from the Ceres solver */
template <class DiffT>
inline Eigen::Matrix<DiffT, 3, 1> QuaternionToAngleAxis(const Eigen::Quaternion<DiffT> quaternion) {
    Eigen::Matrix<DiffT, 3, 1> angle_axis;
    const DiffT& q1 = quaternion.x();
    const DiffT& q2 = quaternion.y();
    const DiffT& q3 = quaternion.z();
    const DiffT sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

    // For quaternions representing non-zero rotation, the conversion
    // is numerically stable.
    if (sin_squared_theta > DiffT(0.0)) {
        const DiffT sin_theta = sqrt(sin_squared_theta);
        const DiffT& cos_theta = quaternion.w();

        // If cos_theta is negative, theta is greater than pi/2, which
        // means that angle for the angle_axis vector which is 2 * theta
        // would be greater than pi.
        //
        // While this will result in the correct rotation, it does not
        // result in a normalized angle-axis vector.
        //
        // In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
        // which is equivalent saying
        //
        //   theta - pi = atan(sin(theta - pi), cos(theta - pi))
        //              = atan(-sin(theta), -cos(theta))
        //
        const DiffT two_theta =
            DiffT(2.0) * ((cos_theta < DiffT(0.0)) ? atan2(-sin_theta, -cos_theta)
                                                   : atan2(sin_theta, cos_theta));
        const DiffT k = two_theta / sin_theta;
        angle_axis.x() = q1 * k;
        angle_axis.y() = q2 * k;
        angle_axis.z() = q3 * k;
    } else {
        // For zero rotation, sqrt() will produce NaN in the derivative since
        // the argument is zero.  By approximating with a Taylor series,
        // and truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used.
        const DiffT k(2.0);
        angle_axis.x() = q1 * k;
        angle_axis.y() = q2 * k;
        angle_axis.z() = q3 * k;
    }
    return angle_axis;
}

template <class DiffT>
inline Eigen::Quaternion<DiffT> AngleAxisToQuaternion(const Eigen::Matrix<DiffT, 3, 1> angle_axis) {
    Eigen::Quaternion<DiffT> quaternion;
    const DiffT& a0 = angle_axis.x();
    const DiffT& a1 = angle_axis.y();
    const DiffT& a2 = angle_axis.z();
    const DiffT theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    // For points not at the origin, the full conversion is numerically stable.
    if (theta_squared > DiffT(0.0)) {
        const DiffT theta = sqrt(theta_squared);
        const DiffT half_theta = theta * DiffT(0.5);
        const DiffT k = sin(half_theta) / theta;
        quaternion.w() = cos(half_theta);
        quaternion.x() = a0 * k;
        quaternion.y() = a1 * k;
        quaternion.z() = a2 * k;
    } else {
        // At the origin, sqrt() will produce NaN in the derivative since
        // the argument is zero.  By approximating with a Taylor series,
        // and truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used.
        const DiffT k(0.5);
        quaternion.w() = DiffT(1.0);
        quaternion.x() = a0 * k;
        quaternion.y() = a1 * k;
        quaternion.z() = a2 * k;
    }
    return quaternion;
}

template <typename T>
Eigen::Matrix<T, 3, 3> AngleAxisToRotationMatrix(const Eigen::Matrix<T, 3, 1> angle_axis) {
    Eigen::Matrix<T, 3, 3> R;
    static const T kOne = T(1.0);
    const T theta2 = angle_axis.dot(angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon())) {
        // We want to be careful to only evaluate the square root if the
        // norm of the angle_axis vector is greater than zero. Otherwise
        // we get a division by zero.
        const T theta = sqrt(theta2);
        const T wx = angle_axis[0] / theta;
        const T wy = angle_axis[1] / theta;
        const T wz = angle_axis[2] / theta;

        const T costheta = cos(theta);
        const T sintheta = sin(theta);

        // clang-format off
    R(0, 0) =     costheta   + wx*wx*(kOne -    costheta);
    R(1, 0) =  wz*sintheta   + wx*wy*(kOne -    costheta);
    R(2, 0) = -wy*sintheta   + wx*wz*(kOne -    costheta);
    R(0, 1) =  wx*wy*(kOne - costheta)     - wz*sintheta;
    R(1, 1) =     costheta   + wy*wy*(kOne -    costheta);
    R(2, 1) =  wx*sintheta   + wy*wz*(kOne -    costheta);
    R(0, 2) =  wy*sintheta   + wx*wz*(kOne -    costheta);
    R(1, 2) = -wx*sintheta   + wy*wz*(kOne -    costheta);
    R(2, 2) =     costheta   + wz*wz*(kOne -    costheta);
        // clang-format on
    } else {
        // Near zero, we switch to using the first order Taylor expansion.
        R(0, 0) = kOne;
        R(1, 0) = angle_axis[2];
        R(2, 0) = -angle_axis[1];
        R(0, 1) = -angle_axis[2];
        R(1, 1) = kOne;
        R(2, 1) = angle_axis[0];
        R(0, 2) = angle_axis[1];
        R(1, 2) = -angle_axis[0];
        R(2, 2) = kOne;
    }
    return R;
}

struct GyroIntegrator {
    typedef Eigen::AutoDiffScalar<Eigen::Vector3d> DiffT;
    typedef Eigen::Quaternion<DiffT> RotT;
    typedef Eigen::Matrix<DiffT, 3, 1> RVT;

    struct BiasedGyroThunk {
        Eigen::Vector3d rot;
        Eigen::Vector3d dt1, dt2;
    };

    struct GyroThunk {
        RVT rot;
        RVT dt1, dt2;

        BiasedGyroThunk Bias(Eigen::Vector3d bias) const {
            Eigen::Vector3d b_rot, b_dt1, b_dt2;
            // clang-format off
            b_rot << 
                rot[0].value() + bias.dot(rot[0].derivatives()),
                rot[1].value() + bias.dot(rot[1].derivatives()),
                rot[2].value() + bias.dot(rot[2].derivatives());
            b_dt1 << 
                dt1[0].value() + bias.dot(dt1[0].derivatives()),
                dt1[1].value() + bias.dot(dt1[1].derivatives()),
                dt1[2].value() + bias.dot(dt1[2].derivatives());
            b_dt2 << 
                dt2[0].value() + bias.dot(dt2[0].derivatives()),
                dt2[1].value() + bias.dot(dt2[1].derivatives()),
                dt2[2].value() + bias.dot(dt2[2].derivatives());
            // clang-format on
            return {b_rot, b_dt1, b_dt2};
        }

        Eigen::Vector3d FindBias(Eigen::Vector3d actual_rotation) const {
            auto e = rot - actual_rotation.cast<DiffT>();
            Eigen::Matrix<double, 3, 1> ea{{e.x().value()}, {e.y().value()}, {e.z().value()}};
            Eigen::Matrix3d jac;
            // clang-format off
            jac << 
                e.x().derivatives()[0], e.x().derivatives()[1], e.x().derivatives()[2],
                e.y().derivatives()[0], e.y().derivatives()[1], e.y().derivatives()[2], 
                e.z().derivatives()[0], e.z().derivatives()[1], e.z().derivatives()[2];
            // clang-format on

            return Eigen::MatrixXd{jac}
                .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(-ea);
        };
    };

    GyroIntegrator() {}

    GyroIntegrator(Eigen::Vector3d* samples, int length) {
        std::vector<RotT> v;
        for (int i = 0; i < length; ++i) {
            RVT rv;
            rv << DiffT{samples[i][0], 3, 0}, DiffT{samples[i][1], 3, 1},
                DiffT{samples[i][2], 3, 2};
            v.push_back(AngleAxisToQuaternion(rv));
        }
        segment_tree_ = {v.begin(), v.end()};
    }

    GyroThunk IntegrateGyro(double t1, double t2) const {
        t1 = std::max(std::min(t1, static_cast<double>(segment_tree_.Size()) - 3.), 1.);
        t2 = std::max(std::min(t2, static_cast<double>(segment_tree_.Size()) - 3.), 1.);

        const int n1 = std::floor(t1);
        const int n2 = std::floor(t2);
        RVT sum1, sum2, dsum1, dsum2;
        {
            const auto p0 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n1 - 1));
            const auto p1 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n1 + 0));
            const auto p2 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n1 + 1));
            const auto p3 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n1 + 2));
            CubicHermiteSpline<3, DiffT>(p0, p1, p2, p3, t1 - static_cast<double>(n1), sum1.data(),
                                         dsum1.data());
        }
        {
            const auto p0 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n2 - 1));
            const auto p1 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n2 + 0));
            const auto p2 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n2 + 1));
            const auto p3 = QuaternionToAngleAxis(segment_tree_.Query(n1 - 1, n2 + 2));
            CubicHermiteSpline<3, DiffT>(p0, p1, p2, p3, t2 - static_cast<double>(n2), sum2.data(),
                                         dsum2.data());
        }
        const RotT sum1aa = AngleAxisToQuaternion(sum1);
        const RotT sum2aa = AngleAxisToQuaternion(sum2);
        auto sumaa = RotT{sum1aa.inverse() * sum2aa};

        return GyroThunk{QuaternionToAngleAxis(sumaa), dsum1, dsum2};
    }

   private:
    /* This function is derived from the same named function in Google's Ceres solver */
    template <int kDataDimension, class T>
    void CubicHermiteSpline(const Eigen::Matrix<T, kDataDimension, 1>& p0,
                            const Eigen::Matrix<T, kDataDimension, 1>& p1,
                            const Eigen::Matrix<T, kDataDimension, 1>& p2,
                            const Eigen::Matrix<T, kDataDimension, 1>& p3, const double x, T* f,
                            T* dfdx) const {
        typedef Eigen::Matrix<T, kDataDimension, 1> VType;
        const VType a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
        const VType b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
        const VType c = 0.5 * (-p0 + p2);
        const VType d = p1;
        // Use Horner's rule to evaluate the function value and its
        // derivative.

        // f = ax^3 + bx^2 + cx + d
        if (f != NULL) {
            Eigen::Map<VType>(f, kDataDimension) = d + x * (c + x * (b + x * a));
        }

        // dfdx = 3ax^2 + 2bx + c
        if (dfdx != NULL) {
            Eigen::Map<VType>(dfdx, kDataDimension) = c + x * (2.0 * b + 3.0 * a * x);
        }
    }

    struct QuaternionGroup {
        typedef Eigen::AutoDiffScalar<Eigen::Vector3d> scalar_type;
        typedef Eigen::Quaternion<scalar_type> value_type;
        value_type unit() const { return value_type::Identity(); }

        value_type add(const value_type& a, const value_type& b) const { return a * b; }

        value_type mult(const value_type& a, double k) const {
            return AngleAxisToQuaternion((QuaternionToAngleAxis(a) * k).eval());
        }

        value_type inv(const value_type& a) const { return mult(a, -1.); }
    };

    SegmentTree<QuaternionGroup> segment_tree_;
};

inline void LowpassGyro(Eigen::Vector3d* samples, int length, int divider) {
    if (divider < 2) return;
    const double ita = 1.0 / tan(M_PI / divider);
    const double q = sqrt(2.0);
    const double b0 = 1.0 / (1.0 + q * ita + ita * ita), b1 = 2 * b0, b2 = b0,
                 a1 = 2.0 * (ita * ita - 1.0) * b0, a2 = -(1.0 - q * ita + ita * ita) * b0;

    Eigen::Vector3d out[3] = {samples[0], samples[1], samples[2]};
    for (int i = 2; i < length; ++i) {
        out[2] = b0 * samples[i] + b1 * samples[i - 1] + b2 * samples[i - 2] + a1 * out[2 - 1] +
                 a2 * out[2 - 2];
        samples[i - 2] = out[0];
        // left shift
        out[0] = out[1];
        out[1] = out[2];
    }
    // reverse pass
    out[0] = samples[length - 1];
    out[1] = samples[length - 2];
    for (int j = 2; j < length; ++j) {
        int i = length - j - 1;
        out[2] = b0 * samples[i] + b1 * samples[i + 1] + b2 * samples[i + 2] + a1 * out[2 - 1] +
                 a2 * out[2 - 2];
        samples[i + 2] = out[0];
        // left shift
        out[0] = out[1];
        out[1] = out[2];
    }
}

inline void UpsampleGyro(Eigen::Vector3d* samples, int length_new, int multiplier) {
    if (multiplier < 2) return;
    int length = length_new / multiplier;
    int half_mult = multiplier / 2;
    int old_samples_base = length_new - length;
    std::copy_n(samples, length, samples + old_samples_base);

    for (int i = 0; i < length_new; ++i) {
        if ((i + half_mult) % multiplier) {
            samples[i] = Eigen::Vector3d::Zero();
        } else {
            samples[i] = samples[i / multiplier + old_samples_base];
        }
    }

    LowpassGyro(samples, length_new, multiplier * 4);
}

inline void DecimateGyro(Eigen::Vector3d* samples, int length, int divider) {
    if (divider < 2) return;
    for (int i = 0; i < length / divider; ++i) {
        samples[i] = samples[i * divider];
    }
}