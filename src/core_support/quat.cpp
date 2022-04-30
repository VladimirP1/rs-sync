#include "quat.hpp"

#include <stdint.h>

arma::vec4 quat_from_aa(arma::vec3 aa) {
    const double theta_squared = arma::dot(aa, aa);

    if (theta_squared > 0.) {
        const double theta = sqrt(theta_squared);
        const double half_theta = theta * 0.5;
        const double k = sin(half_theta) / theta;
        return {cos(half_theta), aa(0) * k, aa(1) * k, aa(2) * k};
    } else {
        const double k(0.5);
        return {1., aa(0) * k, aa(1) * k, aa(2) * k};
    }
}

arma::vec3 quat_to_aa(arma::vec4 q) {
    const auto &xyz = q.rows(1, 3);
    const double sin_squared_theta = arma::dot(xyz, xyz);

    if (sin_squared_theta <= 0.) return xyz * 2;

    const double sin_theta = sqrt(sin_squared_theta);
    const double &cos_theta = q(0);
    const double two_theta =
        2. * ((cos_theta < 0.) ? atan2(-sin_theta, -cos_theta) : atan2(sin_theta, cos_theta));
    const double k = two_theta / sin_theta;
    return xyz * k;
}

arma::vec4 quat_prod(arma::vec4 p, arma::vec4 q) {
    return {(p(0) * q(0) - p(1) * q(1) - p(2) * q(2) - p(3) * q(3)),
            (p(0) * q(1) + p(1) * q(0) + p(2) * q(3) - p(3) * q(2)),
            (p(0) * q(2) - p(1) * q(3) + p(2) * q(0) + p(3) * q(1)),
            (p(0) * q(3) + p(1) * q(2) - p(2) * q(1) + p(3) * q(0))};
}

arma::vec4 quat_conj(arma::vec4 q) {
    q.rows(1, 3) = -q.rows(1, 3);
    return q;
}

arma::vec3 quat_rotate_point(arma::vec4 q, arma::vec3 p) {
    return quat_prod(q, quat_prod({0, p[0], p[1], p[2]}, quat_conj(q))).rows(1,3);
}

static inline arma::vec4 quat_double(arma::vec4 p, arma::vec4 q) {
    return 2. * arma::dot(p, q) * q - p;
}

static inline arma::vec4 quat_bisect(arma::vec4 p, arma::vec4 q) { return (p + q) * .5; }

inline arma::vec4 quat_slerp(arma::vec4 p, arma::vec4 q, double t) {
    if (arma::dot(p, q) < 0) {
        q = -q;
    }

    double mult1, mult2;
    const double theta = acos(arma::dot(p, q));

    // TODO: check if differentiable
    if (theta > 1e-9) {
        const double sin_theta = sin(theta);
        mult1 = sin((1 - t) * theta) / sin_theta;
        mult2 = sin(t * theta) / sin_theta;
    } else {
        mult1 = 1 - t;
        mult2 = t;
    }

    return mult1 * p + mult2 * q;
}

arma::vec4 quat_squad(arma::vec4 p0, arma::vec4 p1, arma::vec4 p2, arma::vec4 p3, double t) {
    arma::vec4 a0 = quat_bisect(quat_double(p0, p1), p2);
    arma::vec4 a1 = quat_bisect(quat_double(p1, p2), p3);
    arma::vec4 b1 = quat_double(a1, p2);
    arma::vec4 &i0 = p1, &i1 = a0, &i2 = b1, &i3 = p2;
    i1 = (i1 + 2 * i0) / 3;
    i2 = (i2 + 2 * i3) / 3;
    arma::vec4 j0 = quat_slerp(i0, i1, t);
    arma::vec4 j1 = quat_slerp(i1, i2, t);
    arma::vec4 j2 = quat_slerp(i2, i3, t);
    return quat_slerp(quat_slerp(j0, j1, t), quat_slerp(j1, j2, t), t);
}

inline arma::vec4 quat_lerp(arma::vec4 p, arma::vec4 q, double t) { return (p * (1 - t) + q * t); }

arma::vec4 quat_quad(arma::vec4 p0, arma::vec4 p1, arma::vec4 p2, arma::vec4 p3, double t) {
    arma::vec4 a0 = quat_bisect(quat_double(p0, p1), p2);
    arma::vec4 a1 = quat_bisect(quat_double(p1, p2), p3);
    arma::vec4 b1 = quat_double(a1, p2);
    a0 = (a0 + 2 * p1) / 3;
    b1 = (b1 + 2 * p2) / 3;
    const arma::vec4 j0 = quat_lerp(p1, a0, t);
    const arma::vec4 j1 = quat_lerp(a0, b1, t);
    const arma::vec4 j2 = quat_lerp(b1, p2, t);
    return quat_lerp(quat_lerp(j0, j1, t), quat_lerp(j1, j2, t), t);
}