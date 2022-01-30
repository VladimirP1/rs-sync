#pragma once
#include "component.hpp"

#include <Eigen/Eigen>

namespace rssync {

class IGyroLoader : public BaseComponent {
   public:
    virtual size_t DataSize() const = 0;

    virtual void GetData(Eigen::Vector3d* ptr, size_t size) const = 0;

    virtual double SampleRate() const = 0;

    virtual void SetOrientation(Eigen::Vector3d orientation) = 0;
};

void RegisterGyroLoader(std::shared_ptr<IContext> ctx, std::string name, std::string filename);

constexpr const char* kGyroLoaderName = "GyroLoader";
}  // namespace rssync