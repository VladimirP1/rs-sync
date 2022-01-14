#pragma once

#include <unordered_map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include <math/prefix_sums.hpp>
#include <math/range_interpolated.hpp>
#include <math/simple_math.hpp>
#include <vision/calibration.hpp>

class MatchContext {
   private:
    int frame_a_{-1}, frame_b_{-1};
    cv::Mat_<double> points_4d_;
    cv::Mat_<double> projection_;
    // Mosaic corr_maps_;
};

class FrameContext {
   private:
    std::vector<std::shared_ptr<MatchContext>> matched_;

};

class SyncContext {
   private:
    FisheyeCalibration calibration_;
    std::unordered_map<int, std::shared_ptr<FrameContext>> frames_;
    Interpolated<PrefixSums<QuaternionGroup>> gyro_integrator_;
};