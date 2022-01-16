#pragma once

#include "component.hpp"

namespace rssync {
void RegisterOpticalFlowLK(std::shared_ptr<IContext> ctx, std::string name);

class IOpticalFlow : public rssync::BaseComponent {
   public:
    virtual bool CalcOptflow(int frame_number) = 0;
};

constexpr const char* kOpticalFlowName = "OpticalFlow";

}  // namespace rssync