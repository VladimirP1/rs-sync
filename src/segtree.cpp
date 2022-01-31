
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

#include <iomanip>

using namespace rssync;

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

    LowpassGyro(rvs.data(), rvs.size(), 10);
    GyroIntegrator interator(rvs.data(), rvs.size());

    double base = 38;
    double duration = 1 / 30.;
    double sweep = 1 / 30.;
    out << std::fixed << std::setprecision(16);
    int count = 0;
    {
        Stopwatch w{"integration"};
        for (double start = base; start < sweep + base; start += .00001) {
            auto res = interator.IntegrateGyro(start * samplerate, (start + duration) * samplerate);
            auto ds = res.dt2 - res.dt1;
            out << start << "," << ds[0] << "," << res.rot[0] * 180 / M_PI << "\n"; 
            ++count;
        }
    }
    std::cout << count << std::endl;

    // double actual_start, actual_step;
    // std::vector<Eigen::Vector3d> raw_rvs;
    // raw_rvs.resize(5001);
    // gyro_loader->GetRawRvs(raw_rvs.size() / 2, 38., actual_start, actual_step, raw_rvs.data());

    // std::vector<Quaternion<double>> quats;

    // ceres::Grid1D<double, 3> grid{reinterpret_cast<double*>(raw_rvs.data()), 0,
    //                               static_cast<int>(raw_rvs.size())};
    // ceres::CubicInterpolator<ceres::Grid1D<double, 3>> interp(grid);
    // double x[3];
    // interp.Evaluate(.5, x);
    // std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
    // grid.GetValue(0, x);
    // std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;

    // double base = atoi(argv[1]);
    // int count = 0;
    // {
    //     Stopwatch w;
    //     for (double ofs = 0; ofs < 1 / 30.; ofs += .00001) {
    //         interp.Evaluate(ofs / actual_step, x);
    //         ++count;
    //         // out << ofs << "," << rv.x() << "," << rv.y() << "," << rv.z() << std::endl;
    //         // auto R = gyro_loader->GetRotation(base + ofs, base + duration + ofs);
    //         // auto rv = R.ToRotationVector();
    //         out << ofs << "," << x[0] << "," << x[1] << "," << x[2] << std::endl;
    //     };
    //     std::cout << count << std::endl;
    // }

    return 0;
}