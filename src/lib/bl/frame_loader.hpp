#pragma once
#include "component.hpp"

#include <memory>
#include <string>

#include <opencv2/core.hpp>


namespace rssync {
struct LoadFrameTaskMessage : public FrameLoaderTaskMessage {
    LoadFrameTaskMessage(int frame) : frame_{frame} {}

    int Frame() { return frame_; }

    std::string ToString() const override { return "[Task] Load frame " + std::to_string(frame_); }

   private:
    int frame_;
};

struct LoadResultMessage : public FrameLoaderEventMessage {
    LoadResultMessage(int frame, double timestamp, cv::Mat image)
        : frame_{frame}, timestamp_{timestamp}, image_{image} {}

    int Frame() { return frame_; }

    double Timestamp() { return timestamp_; }

    cv::Mat Image() { return image_; }

    std::string ToString() const override {
        return "[Event] Frame " + std::to_string(frame_) + " at " + std::to_string(timestamp_) +
               " loaded";
    }

   private:
    int frame_;
    double timestamp_;
    cv::Mat image_;
};

void RegisterFrameLoader(std::shared_ptr<IContext> ctx, std::string name, size_t max_queue, std::string filename);
class FrameLoader : public BaseComponent {
   public:
    virtual void ContextLoaded(std::string name, std::weak_ptr<BaseComponent> self,
                               std::weak_ptr<IContext> ctx);

   private:
};

constexpr const char * kFrameLoaderName = "FrameLoader";
}  // namespace rssync