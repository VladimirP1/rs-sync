#include <iostream>

#include <math/gyro_integrator.hpp>

using Eigen::AngleAxis;
using Eigen::AutoDiffScalar;
using Eigen::Matrix;
using Eigen::Quaternion;
using Eigen::Vector3d;
using std::cout;

int main() {
    std::vector<Eigen::Vector3d> data;

    for (int i = 0; i < 10000; ++i) {
        Vector3d v;
        v << 1e-9, 0, 0;
        data.push_back(v);
    }

    GyroIntegrator gi(data.data(), data.size());

    auto res = gi.IntegrateGyro(4, 1500.3);

    Vector3d bias;
    bias << 1500 * 1e-9, 0, 0;
    auto bres = res.Bias(bias);

    auto bb = res.FindBias(bias);
    std::cout << bb.norm() << std::endl;

    std::cout << res.rot << "\n\n" << res.dt1 << std::endl;
    std::cout << "\n\n";
    std::cout << bres.rot << "\n\n" << bres.dt2 << std::endl;
    // std::cout << q.norm().derivatives()[1] << std::endl;

    return 0;
}