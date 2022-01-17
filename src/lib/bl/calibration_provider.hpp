#pragma once

#include "component.hpp"

#include <vision/calibration.hpp>

namespace rssync {
void RegisterCalibrationProvider(std::shared_ptr<IContext> ctx, std::string name, std::string filename = "");

class ICalibrationProvider : public rssync::BaseComponent {
   public:
    virtual void SetCalibration(const FisheyeCalibration& calibration) = 0;
    virtual FisheyeCalibration GetCalibraiton() = 0;
    virtual cv::Mat GetReasonableProjection() = 0;
};

constexpr const char* kCalibrationProviderName = "CalibrationProvider";

}  // namespace rssync