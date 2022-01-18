#pragma once

#include "component.hpp"

#include <opencv2/core.hpp>

namespace rssync {
void RegisterVisualizer(std::shared_ptr<IContext> ctx, std::string name);

class IVisualizer : public rssync::BaseComponent {
   public:
    virtual void DimImage(cv::Mat& frame, double k) = 0;
    virtual void OverlayMatched(cv::Mat& frame, int frame_number, bool ab,
                                bool undistorted = false) = 0;
    virtual void OverlayMatchedTracks(cv::Mat& frame, int frame_number,
                                      bool undistorted = false) = 0;
};

constexpr const char* kVisualizerName = "Visualizer";

}  // namespace rssync