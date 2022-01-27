
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

using namespace rssync;

int main(int argc, char** argv) {
    std::ofstream out("out.csv");

    auto ctx = IContext::CreateContext();
    RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA_fixed.CSV");
    ctx->ContextLoaded();

    auto gyro_loader = ctx->GetComponent<IGyroLoader>(kGyroLoaderName);

    double base = atoi(argv[1]);
    double duration = 1 / 30.;

    for (double ofs = 0; ofs < 1 / 30.; ofs += .0001) {
        auto R = gyro_loader->GetRotation(base + ofs, base + duration + ofs);
        auto rv = R.ToRotationVector();
        out << ofs << "," << rv.x().a << "," << rv.y().a << "," << rv.z().a << std::endl;
    };

    return 0;
}