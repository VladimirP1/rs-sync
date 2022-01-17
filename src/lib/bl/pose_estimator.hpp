#pragma once

#include "component.hpp"

namespace rssync {
void RegisterPoseEstimator(std::shared_ptr<IContext> ctx, std::string name);

class IPoseEstimator : public rssync::BaseComponent {
   public:
    virtual bool EstimatePose(int frame_number) = 0;
};

constexpr const char* kPoseEstimatorName = "PoseEstimator";

}  // namespace rssync