#pragma once
#include "component.hpp"

#include <math/simple_math.hpp>

namespace rssync {

class IGyroLoader : public BaseComponent {
   public:
    virtual Quaternion GetRotation(double from_sec, double to_sec) const = 0;
};

void RegisterGyroLoader(std::shared_ptr<IContext> ctx, std::string name, 
                         std::string filename);

constexpr const char* kGyroLoaderName = "GyroLoader";
}  // namespace rssync