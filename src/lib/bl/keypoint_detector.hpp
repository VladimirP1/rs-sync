#pragma once
#include "component.hpp"

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rssync {
struct DetectKeypointsTaskMessage : public KeypointDetectorTaskMessage {
    DetectKeypointsTaskMessage(int frame) : frame_{frame} {}

    int Frame() { return frame_; }

    std::string ToString() const override {
        return "[Task] Detect keypoints in frame " + std::to_string(frame_);
    }

   private:
    int frame_;
    cv::Mat image_;
};

struct KeypointsDetectedMessage : public KeypointDetectorEventMessage {
    KeypointsDetectedMessage(int frame, std::vector<cv::Point2f> pts)
        : frame_{frame}, pts_{std::move(pts)} {}

    int Frame() { return frame_; }

    const std::vector<cv::Point2f>& Keypoints() { return pts_; }

    std::string ToString() const override {
        return "[Event] Detected " + std::to_string(pts_.size()) + " keypoints in frame " +
               std::to_string(frame_);
    }

   private:
    int frame_;
    std::vector<cv::Point2f> pts_;
};

std::shared_ptr<BaseComponent> RegisterKeypointDetector(std::shared_ptr<IContext> ctx, std::string name, size_t max_queue);

constexpr const char * kKeypointDetectorName = "KeypointDetector";


}  // namespace rssync