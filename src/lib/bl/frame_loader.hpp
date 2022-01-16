#pragma once
#include "component.hpp"

#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace rssync {

void RegisterFrameLoader(std::shared_ptr<IContext> ctx, std::string name, 
                         std::string filename);
class IFrameLoader : public BaseComponent {
   public:
    virtual bool GetFrame(int n, cv::Mat& out) = 0;

   private:
};

constexpr const char* kFrameLoaderName = "FrameLoader";
}  // namespace rssync