#pragma once
#include "component.hpp"

namespace rssync {

class IRoughGyroCorrelator : public rssync::BaseComponent {
   public:
    virtual void Run(double search_radius, double search_step, int start_frame, int end_frame) = 0;
};

void RegisterRoughGyroCorrelator(std::shared_ptr<IContext> ctx, std::string name);

constexpr const char* kRoughGyroCorrelatorName = "RoughGyroCorrelator";
}  // namespace rssync