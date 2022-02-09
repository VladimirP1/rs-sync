#pragma once
#include "component.hpp"

#include <Eigen/Eigen>

#include <vector>

namespace rssync {

class IFineSync : public rssync::BaseComponent {
   public:
    virtual double Run(double initial_offset, Eigen::Vector3d bias, double search_radius, double search_step, int start_frame, int end_frame) = 0;
    virtual double Run2(double initial_offset, Eigen::Vector3d bias, double search_radius, double search_step, int start_frame, int end_frame) = 0;
};

void RegisterFineSync(std::shared_ptr<IContext> ctx, std::string name);

constexpr const char* kFineSyncName = "FineSync";
}  // namespace rssync