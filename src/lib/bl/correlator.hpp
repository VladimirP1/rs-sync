#pragma once
#include "component.hpp"

#include <opencv2/core.hpp>

namespace rssync {
void RegisterCorrelator(std::shared_ptr<IContext> ctx, std::string name);

class ICorrelator : public rssync::BaseComponent {
   public:
    virtual void SetPatchSizes(cv::Size dst_a, cv::Size dst_b) = 0;
    virtual bool RefineOF(int frame_number) = 0;
};

constexpr const char* kCorrelatorName = "Correlator";

}  // namespace rssync