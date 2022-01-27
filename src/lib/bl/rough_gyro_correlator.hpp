#pragma once
#include "component.hpp"

#include <math/quaternion.hpp>
namespace rssync {

struct RoughCorrelationReport {
    double offset{};
    Matrix<double, 3, 1> bias_estimate{};
    std::vector<int> frames;
};

class IRoughGyroCorrelator : public rssync::BaseComponent {
   public:
    virtual void Run(double initial_offset, double search_radius, double search_step, int start_frame, int end_frame, RoughCorrelationReport* report = nullptr) = 0;
};

void RegisterRoughGyroCorrelator(std::shared_ptr<IContext> ctx, std::string name);

constexpr const char* kRoughGyroCorrelatorName = "RoughGyroCorrelator";
}  // namespace rssync