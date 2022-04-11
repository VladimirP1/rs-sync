#pragma once

#include "cv.hpp"
#include "ndspline.hpp"

struct OptData {
    double quats_start{};
    int sample_rate{};
    ndspline quats{};

    FramesFlow flows{};
};