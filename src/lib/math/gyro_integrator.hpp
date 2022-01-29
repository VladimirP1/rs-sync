#pragma once

#include <ds/segment_tree.hpp>

#include <Eigen/Eigen>
#include <unsupported/Eigen/AutoDiff>

inline void LowpassGyro(Eigen::Vector3d* samples, int length, double divider) {
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

struct GyroIntegrator {
    typedef Eigen::AutoDiffScalar<Eigen::Vector3d> DiffT;
    typedef Eigen::AngleAxis<DiffT> RotT;
    typedef Eigen::Matrix<DiffT, 3, 1> RVT;

    struct BiasedGyroThunk {
        const Eigen::Vector3d rot;
        const Eigen::Vector3d dt1, dt2;
    };

    struct GyroThunk {
        const RVT rot;
        const RVT dt1, dt2;

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

    GyroIntegrator(Eigen::Vector3d* samples, int length) {
        std::vector<RotT> v;
        for (int i = 0; i < length; ++i) {
            RVT rv;
            rv << DiffT{samples[i][0], 3, 0}, DiffT{samples[i][1], 3, 1},
                DiffT{samples[i][2], 3, 2};
            v.push_back({rv.norm(), rv.normalized()});
        }
        segment_tree_ = {v.begin(), v.end()};
    }

    GyroThunk IntegrateGyro(double t1, double t2) {
        const int n1 = std::floor(t1);
        const int n2 = std::floor(t2);
        RVT sum1, sum2, dsum1, dsum2;
        {
            const auto p0 = segment_tree_.Query(n1 - 1, n1 - 1);
            const auto p1 = segment_tree_.Query(n1 - 1, n1 + 0);
            const auto p2 = segment_tree_.Query(n1 - 1, n1 + 1);
            const auto p3 = segment_tree_.Query(n1 - 1, n1 + 2);
            CubicHermiteSpline<3, DiffT>(p0.angle() * p0.axis(), p1.angle() * p1.axis(),
                                         p2.angle() * p2.axis(), p3.angle() * p3.axis(),
                                         t1 - static_cast<double>(n1), sum1.data(), dsum1.data());
        }
        {
            const auto p0 = segment_tree_.Query(n1 - 1, n2 - 1);
            const auto p1 = segment_tree_.Query(n1 - 1, n2 + 0);
            const auto p2 = segment_tree_.Query(n1 - 1, n2 + 1);
            const auto p3 = segment_tree_.Query(n1 - 1, n2 + 2);
            CubicHermiteSpline<3, DiffT>(p0.angle() * p0.axis(), p1.angle() * p1.axis(),
                                         p2.angle() * p2.axis(), p3.angle() * p3.axis(),
                                         t2 - static_cast<double>(n2), sum2.data(), dsum2.data());
        }
        const RotT sum1aa{sum1.norm(), sum1.normalized()};
        const RotT sum2aa{sum2.norm(), sum2.normalized()};
        auto sumaa = RotT{sum1aa.inverse() * sum2aa};

        return GyroThunk{sumaa.axis() * sumaa.angle(), dsum1, dsum2};
    }

   private:
    template <int kDataDimension, class T>
    void CubicHermiteSpline(const Eigen::Matrix<T, kDataDimension, 1>& p0,
                            const Eigen::Matrix<T, kDataDimension, 1>& p1,
                            const Eigen::Matrix<T, kDataDimension, 1>& p2,
                            const Eigen::Matrix<T, kDataDimension, 1>& p3, const double x, T* f,
                            T* dfdx) {
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

    struct AngleAxisGroup {
        typedef Eigen::AutoDiffScalar<Eigen::Vector3d> scalar_type;
        typedef Eigen::AngleAxis<scalar_type> value_type;
        value_type unit() const { return value_type::Identity(); }

        value_type add(const value_type& a, const value_type& b) const { return value_type{a * b}; }

        value_type mult(const value_type& a, double k) const {
            return value_type{a.angle() * k, a.axis()};
        }

        value_type inv(const value_type& a) const { return value_type{-a.angle(), a.axis()}; }
    };

    SegmentTree<AngleAxisGroup> segment_tree_;
};