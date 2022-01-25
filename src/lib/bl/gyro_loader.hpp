#pragma once
#include "component.hpp"

#include <math/quaternion.hpp>

namespace rssync {

class IGyroLoader : public BaseComponent {
   public:
    using QuatT = Quaternion<ceres::Jet<double, 3>>;
    virtual void SetOrientation(Quaternion<double> orient) = 0;
    virtual QuatT GetRotation(double from_sec, double to_sec) const = 0;
};

void RegisterGyroLoader(std::shared_ptr<IContext> ctx, std::string name, std::string filename);

constexpr const char* kGyroLoaderName = "GyroLoader";
}  // namespace rssync