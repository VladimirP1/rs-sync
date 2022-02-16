
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



int main(int argc, char** argv) {
    std::ofstream out("log.csv");
    std::ifstream in("GX019642.MP4.csv");

    std::vector<double> timestamps;
    std::vector<Eigen::Vector3d> rvs;
    ReadGyroCsv(in, timestamps, rvs);

    double samplerate = timestamps.size() / (timestamps.back() - timestamps.front());

    for (auto& rv : rvs) {
        rv /= samplerate;
    }

    LowpassGyro(rvs.data(), rvs.size(), 5);

    double sobel[3] = {-1, 0, 1};
    Convolve(rvs[0].data(), sobel, rvs.size(), 3, 3);

    // double gaussian[66];
    // MakeGaussianKernel(gaussian, 66, 33.);
    // Convolve(rvs[0].data(), gaussian, rvs.size(), 3, 66);
    LowpassGyro(rvs.data(), rvs.size(), 10);

    std::vector<double> sync_qual;
    for (int i = 0; i < rvs.size(); ++i) {
        auto rv = rvs[i].array() * rvs[i].array();
        sync_qual.push_back(rv.x() * rv.y() + rv.y() * rv.z() + rv.z() * rv.x());
        // out << std::fixed << std::setprecision(16) << i << "," << rvs[i].x() << std::endl;
        // out << std::fixed << std::setprecision(16) << i << "," << sync_qual.back() << std::endl;
    }

    NonMaxSupress(sync_qual.data(), sync_qual.size(), 1, 1000);

    for (int i = 0; i < sync_qual.size(); ++i) {
        out << std::fixed << std::setprecision(16) << i << "," << sync_qual[i] << std::endl;
        if (sync_qual[i] > 0) {
            std::cout << timestamps[i] << std::endl;
        }
    }
    return 0;
}