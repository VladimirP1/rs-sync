#include "frame_cache.hpp"

#include <opencv2/videoio.hpp>

#include <ds/lru_cache.hpp>

#include <iostream>

class FrameLoaderImpl : public FrameLoader {
    static constexpr size_t kCacheSize = 128;

   public:
    FrameLoaderImpl(MessageQueuePtr queue, std::string filename);

    void Run() override;

   private:
    void ProcessTask(FrameLoaderTaskMessage* msg);

   private:
    int cur_frame_{-1};
    double fps_{1};
    MessageQueueT::Subscription subscription_;
    MessageQueuePtr message_queue_;
    cv::VideoCapture cap_;
    LruCache<int, std::pair<double, cv::Mat>> cache_{kCacheSize};
};

FrameLoaderImpl::FrameLoaderImpl(MessageQueuePtr queue, std::string filename)
    : message_queue_{queue}, cap_{filename} {
    subscription_ = message_queue_->Subscribe();
    fps_ = cap_.get(cv::CAP_PROP_FPS);
}

void FrameLoaderImpl::Run() {
    std::shared_ptr<Message> message;
    while (subscription_.Dequeue(message)) {
        if (auto task = dynamic_cast<FrameLoaderTaskMessage*>(message.get()); task) {
            ProcessTask(task);
        } else if (auto task = dynamic_cast<FrameLoaderEventMessage*>(message.get()); task) {
        }
    }
}

void FrameLoaderImpl::ProcessTask(FrameLoaderTaskMessage* msg) {
    if (auto load_message = dynamic_cast<LoadFrameTaskMessage*>(msg);
        load_message && load_message->Take()) {
        auto desired_frame = load_message->Frame();
        if (auto maybe_frame = cache_.get(desired_frame); maybe_frame) {
            auto& [ts, frame] = maybe_frame.value();
            message_queue_->Enqueue(
                std::make_shared<LoadResultMessage>(desired_frame, ts, frame));
        } else {
            cv::Mat frame;
            if (desired_frame != cur_frame_) {
                cap_.set(cv::CAP_PROP_POS_FRAMES, desired_frame);
            }
            double cur_timestamp_ = desired_frame / fps_;
            if (cap_.read(frame)) {
                message_queue_->Enqueue(std::make_shared<LoadResultMessage>(desired_frame, cur_timestamp_, frame));
                cache_.put(desired_frame, {cur_timestamp_, frame});
                cur_frame_ = desired_frame + 1;
            } else {
                message_queue_->Enqueue(std::make_shared<LoadResultMessage>(-1, 0., frame));
                cur_frame_ = -1;
            }
        }
    }
}

std::shared_ptr<FrameLoader> FrameLoader::Create(MessageQueuePtr queue, std::string filename) {
    return std::make_shared<FrameLoaderImpl>(queue, filename);
}

FrameLoader::~FrameLoader() {}