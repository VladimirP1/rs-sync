#pragma once
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include <bl/message_types.hpp>

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
        return "[Event] Frame " + std::to_string(frame_) + " at " + std::to_string(timestamp_) + " loaded";
    }

   private:
    int frame_;
    double timestamp_;
    cv::Mat image_;
};

class FrameLoader {
   public:
    FrameLoader() {}
    FrameLoader(const FrameLoader&) = delete;
    FrameLoader(FrameLoader&&) = delete;
    FrameLoader& operator=(const FrameLoader&) = delete;
    FrameLoader& operator=(FrameLoader&&) = delete;

    static std::shared_ptr<FrameLoader> Create(MessageQueuePtr queue, std::string filename);

    virtual void Run() = 0;

    virtual ~FrameLoader();
};