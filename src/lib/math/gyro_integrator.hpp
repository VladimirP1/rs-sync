#pragma once

#include "rotation.h"

#include <ds/segment_tree.hpp>
// #include <ds/range_query_cache.hpp>

#include <Eigen/Eigen>
#include <unsupported/Eigen/AutoDiff>

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

        GyroThunk Mix(const GyroThunk& b, double k) const {
            return GyroThunk{(1 - k) * rot + k * b.rot, (1 - k) * dt1 + k * b.dt1,
                             (1 - k) * dt2 + k * b.dt2};
        }
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
        const RotT sum1aa = AngleAxisToQuaternion((-sum1).eval());
        const RotT sum2aa = AngleAxisToQuaternion(sum2);
        auto sumaa = RotT{sum1aa * sum2aa};

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

    // RangeQueryCache<SegmentTree<QuaternionGroup>> segment_tree_;
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

inline void Convolve(double* data, double* kernel, int nsamples, int ndim, int ksize) {
    std::vector<double> newdata(ndim * nsamples);
    for (int dim = 0; dim < ndim; ++dim) {
        for (int i = 0; i < nsamples; ++i) {
            for (int j = std::max(0, i - ksize / 2); j < std::min(nsamples, i + ksize / 2); ++j) {
                newdata[ndim * i + dim] += data[ndim * j + dim] * kernel[ksize / 2 + j - i];
            }
        }
    }
    std::copy(newdata.begin(), newdata.end(), data);
}

inline void MakeGaussianKernel(double* kernel, int ksize, double sigma) {
    for (int i = 0; i < ksize; ++i) {
        double x = i - static_cast<int>(ksize / 2);
        kernel[i] = exp(-x * x / (sigma * sigma) / 2) / sigma / sqrt(2 * M_PI);
    }
}

inline void NonMaxSupress(double* data, int nsamples, int ndim, int radius) {
    std::vector<double> newdata(ndim * nsamples);
    for (int dim = 0; dim < ndim; ++dim) {
        for (int i = 0; i < nsamples; ++i) {
            double cur = data[ndim * i + dim];
            newdata[ndim * i + dim] = cur;
            for (int j = std::max(0, i - radius); j < std::min(nsamples, i + radius); ++j) {
                if (data[ndim * j + dim] > cur) {
                    newdata[ndim * i + dim] = 0;
                }
            }
        }
    }
    std::copy(newdata.begin(), newdata.end(), data);
}

inline std::vector<int> SuggestSyncPoints(double* data, int nsamples, int lpf1=20, int lpf2=100, int radius=5000) {
    std::vector<int> sync_pts;

    LowpassGyro(reinterpret_cast<Eigen::Vector3d*>(data), nsamples, lpf1);

    double sobel[3] = {-1, 0, 1};
    Convolve(data, sobel, nsamples, 3, 3);

    LowpassGyro(reinterpret_cast<Eigen::Vector3d*>(data), nsamples, lpf2);

    std::vector<double> sync_qual;
    for (int i = 0; i < nsamples; ++i) {
        Eigen::Vector3d rv;
        rv << data[3 * i], data[3 * i + 1], data[3 * i + 2];
        sync_qual.push_back(rv.x() * rv.y() + rv.y() * rv.z() + rv.z() * rv.x());
    }

    NonMaxSupress(sync_qual.data(), sync_qual.size(), 1, radius);

    for (int i = 0; i < sync_qual.size(); ++i) {
        if (sync_qual[i] > 0) {
            sync_pts.push_back(i);
        }
    }

    return sync_pts;
}