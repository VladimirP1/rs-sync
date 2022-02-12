#pragma once
#include <Eigen/Eigen>

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