
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

#include <bl/context.hpp>
#include <io/bb_csv.hpp>

#include <io/stopwatch.hpp>

#include <math/gyro_integrator.hpp>
#include <ds/prefix_sums.hpp>

#include <iomanip>

#include <ceres/cubic_interpolation.h>

using namespace rssync;

struct Integrator {
    using Scalar = Eigen::AutoDiffScalar<Eigen::Vector3d>;
    using Vector3ad = Eigen::Matrix<Scalar, 3, 1>;
    using RetScalar = Eigen::AutoDiffScalar<Eigen::Matrix<double, 4, 1>>;
    using RetAd = Eigen::Matrix<RetScalar, 3, 1>;
    Integrator() {}
    Integrator(Eigen::Vector3d* rvs, size_t size) : intz_(size), size_(size) {
        std::vector<Eigen::Quaternion<Scalar>> as_quat(size);
        for (int i = 0; i < size; ++i) {
            intz_[i] << Scalar{rvs[i].x(), 3, 0}, Scalar{rvs[i].y(), 3, 1},
                Scalar{rvs[i].z(), 3, 2};
            as_quat[i] = AngleAxisToQuaternion(intz_[i]);
        }
        for (size_t i = 1; i < size; ++i) {
            as_quat[i] = as_quat[i - 1] * as_quat[i];
            intz_[i] = QuaternionToAngleAxis(as_quat[i]);
        }
    }

    Vector3ad Integrate(double l, double r, Eigen::Vector3d bias) const {
        l = std::max(std::min(l, size_ - 3.), 1.);
        r = std::max(std::min(r, size_ - 3.), 1.);

        const int nl = std::floor(l);
        const int nr = std::floor(r);
        Vector3ad suml, sumr, dsuml, dsumr;
        {
            const auto p0 = intz_[nl - 1];
            const auto p1 = intz_[nl + 0];
            const auto p2 = intz_[nl + 1];
            const auto p3 = intz_[nl + 2];
            CubicHermiteSpline<3, Scalar>(p0, p1, p2, p3, l - static_cast<double>(nl), suml.data(),
                                          dsuml.data());
        }
        {
            const auto p0 = intz_[nr - 1];
            const auto p1 = intz_[nr + 0];
            const auto p2 = intz_[nr + 1];
            const auto p3 = intz_[nr + 2];
            CubicHermiteSpline<3, Scalar>(p0, p1, p2, p3, r - static_cast<double>(nr), sumr.data(),
                                          dsumr.data());
        }
        const auto sumlaa = AngleAxisToQuaternion((-suml).eval());
        const auto sumraa = AngleAxisToQuaternion(sumr);
        auto sumaa = QuaternionToAngleAxis(sumlaa * sumraa);

        return sumaa;
    }

   private:
    // static size_t tri(size_t l, size_t r) { return l + (r + 1) * r / 2; }

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
        if (f != NULL) Eigen::Map<VType>(f, kDataDimension) = d + x * (c + x * (b + x * a));
        if (dfdx != NULL) Eigen::Map<VType>(dfdx, kDataDimension) = c + x * (2.0 * b + 3.0 * a * x);
    }

    std::vector<Vector3ad> intz_;
    size_t size_;
};

int main(int argc, char** argv) {
    std::ofstream out("log.csv");
    std::ifstream in("000458AA_fixed.CSV");

    std::vector<double> timestamps;
    std::vector<Eigen::Vector3d> rvs;
    ReadGyroCsv(in, timestamps, rvs);

    double samplerate = timestamps.size() / (timestamps.back() - timestamps.front());

    for (auto& rv : rvs) {
        rv /= samplerate;
    }

    // LowpassGyro(rvs.data(), rvs.size(), 10);

    Integrator I(rvs.data() + 32 * 1000, 1000);

    out << std::fixed << std::setprecision(16);
    for (double i = 100; i < 110; i += 1. / 1000) {
        auto ii = I.Integrate(i, i + 224./100, {0.001, 0, 0});
        out << i << "," << ii.x() << "," << ii.x().derivatives()(0, 0) << std::endl;
    }

    return 0;
}