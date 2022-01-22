
static inline Quaternion FromRotationVector(const Eigen::Matrix<Scalar, 3, 1> &rv) {
    auto norm = rv.norm();
    auto a = ceres::cos(norm / 2.);
    auto k = SinxInvx(norm / 2.) / 2.;
    auto b = rv.x() * k;
    auto c = rv.y() * k;
    auto d = rv.z() * k;
    if (Real(norm) < eps) {
        a = Scalar{1.};
        b = rv.x() / 2.;
        c = rv.y() / 2.;
        d = rv.z() / 2.;
    }
    return Quaternion{a, b, c, d};
}

Eigen::Matrix<Scalar, 3, 1> ToRotationVector() const {
    Eigen::Matrix<Scalar, 3, 1> rv;
    auto& q = *this;
    auto cos = q.w();
    auto sin_norm = ceres::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    auto angle = 2. * ceres::atan2(sin_norm, cos);
    if (Real(sin_norm) < eps) {
        rv.x() = rv.y() = rv.z() = {};
        return rv;
    }
    rv.x() = q.x() / sin_norm * angle;
    rv.y() = q.y() / sin_norm * angle;
    rv.z() = q.z() / sin_norm * angle;
    return rv;
}

private:
static constexpr double eps = 1e-15;

template <typename Q>
static inline double Real(const Q& x) {
    return x;
}

template <typename Q, int N>
static inline Q Real(const ceres::Jet<Q, N>& x) {
    return x.a;
}

template <class Q>
static inline Q SinxInvx(Q x) {
    if (fabs(Real(x)) < eps) {
        return -x * x / 6.;
    }
    return ceres::sin(x) / x;
}