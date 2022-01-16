#pragma once
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <bl/message_types.hpp>

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

class KeypointDetector {
   public:
    KeypointDetector() {}
    KeypointDetector(const KeypointDetector&) = delete;
    KeypointDetector(KeypointDetector&&) = delete;
    KeypointDetector& operator=(const KeypointDetector&) = delete;
    KeypointDetector& operator=(KeypointDetector&&) = delete;

    static std::shared_ptr<KeypointDetector> Create(MessageQueuePtr queue);

    virtual void Run() = 0;

    virtual ~KeypointDetector();
};