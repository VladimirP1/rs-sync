
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

#include <bl/context.hpp>
#include <bl/gyro_loader.hpp>

#include <math/quaternion.hpp>

#include <io/stopwatch.hpp>

using namespace rssync;

Eigen::Vector3d GetGyroDerivative(IGyroLoader* loader, double time, double enlarge) {
    class CubicBcKernel {
       public:
        CubicBcKernel(double B = 0., double C = .5)
            : P0{(6. - 2. * B) / 6.},
              P1{0.},
              P2{(-18. + 12. * B + 6. * C) / 6.},
              P3{(12. - 9. * B - 6. * C) / 6.},
              Q0{(8. * B + 24. * C) / 6.},
              Q1{(-12. * B - 48. * C) / 6.},
              Q2{(6. * B + 30. * C) / 6.},
              Q3{(-1. * B - 6. * C) / 6.} {}

        double operator()(double x) const {
            if (x < 0) x = -x;
            if (x < 1.) return P0 + P1 * x + P2 * x * x + P3 * x * x * x;
            if (x < 2.) return Q0 + Q1 * x + Q2 * x * x + Q3 * x * x * x;
            return 0.;
        }

       private:
        double P0, P1, P2, P3, Q0, Q1, Q2, Q3;
    };
    static const CubicBcKernel krnl;

    double act_start, act_step;

    Eigen::Vector3d rvs[128];
    int d = std::ceil(5 * enlarge);
    if (d > 128) {
        abort();
    }
    loader->GetRawRvs(d / 2, time, act_start, act_step, rvs);

    Eigen::Vector3d rv{0, 0, 0};
    for (int i = 0; i < d; ++i) {
        double k = krnl((act_start + i * act_step - time) / act_step / enlarge);
        rv += rvs[i] * k;
    }
    return rv;
}

int main(int argc, char** argv) {
    std::ofstream out("out.csv");

    auto ctx = IContext::CreateContext();
    RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA_fixed.CSV");
    ctx->ContextLoaded();

    auto gyro_loader = ctx->GetComponent<IGyroLoader>(kGyroLoaderName);

    QuaternionGroup<Quaternion<double>> grp;

    double base = atoi(argv[1]);
    double duration = 1 / 30.;
    double krnl_enlarge = atof(argv[2]);
    int count = 0;

    {
        Stopwatch w;
        for (double ofs = 0; ofs < 1 / 30.; ofs += .00001) {
            auto rv = GetGyroDerivative(gyro_loader.get(), base + ofs, krnl_enlarge);
            ++count;
            // out << ofs << "," << rv.x() << "," << rv.y() << "," << rv.z() << std::endl;
            // auto R = gyro_loader->GetRotation(base + ofs, base + duration + ofs);
            // auto rv = R.ToRotationVector();
            // out << ofs << "," << rv.x().a << "," << rv.y().a << "," << rv.z().a << std::endl;
        };
        std::cout << count << std::endl;
    }

    return 0;
}