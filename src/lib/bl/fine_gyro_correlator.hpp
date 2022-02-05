#pragma once
#include "component.hpp"

#include <Eigen/Eigen>

#include <vector>

namespace rssync {

class IFineGyroCorrelator : public rssync::BaseComponent {
   public:
    virtual double Run(double initial_offset, double search_radius, double search_step, int start_frame, int end_frame) = 0;
};

void RegisterFineGyroCorrelator(std::shared_ptr<IContext> ctx, std::string name);

constexpr const char* kFineGyroCorrelatorName = "FineGyroCorrelator";
}  // namespace rssync