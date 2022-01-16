#include "keypoint_detector.hpp"
#include "frame_loader.hpp"

#include <set>

#include <opencv2/video/tracking.hpp>

#include <iostream>

class KeypointDetectorImpl : public KeypointDetector {
   public:
    KeypointDetectorImpl(MessageQueuePtr queue);

    void Run() override;

   private:
    void SendFrameRequestForTask(std::shared_ptr<Message>);

    void ProcessEvent(FrameLoaderEventMessage* msg);

   private:
    MessageQueueT::Subscription subscription_;
    MessageQueuePtr message_queue_;

    std::deque<std::shared_ptr<Message>> tasks_;
    std::deque<std::shared_ptr<Message>> events_;
    std::set<int> taken_frames_;

    int min_corners_{70};
    int max_corners_{700};
    double discard_threshold_{1e-3};
};

KeypointDetectorImpl::KeypointDetectorImpl(MessageQueuePtr queue) : message_queue_{queue} {
    subscription_ = message_queue_->Subscribe();
}

void KeypointDetectorImpl::SendFrameRequestForTask(std::shared_ptr<Message> msg) {
    if (auto task = dynamic_cast<DetectKeypointsTaskMessage*>(msg.get()); task && task->Take()) {
        taken_frames_.insert(task->Frame());
        message_queue_->Enqueue(std::make_shared<LoadFrameTaskMessage>(task->Frame()));
    }
}

void KeypointDetectorImpl::ProcessEvent(FrameLoaderEventMessage* msg) {
    if (auto event = dynamic_cast<LoadResultMessage*>(msg); event && event->Take()) {
        cv::Mat img;
        cv::cvtColor(event->Image(), img, cv::COLOR_BGR2GRAY);

        std::cout << img.cols << std::endl;

        std::vector<cv::Point2f> corners;
        int minDist = std::sqrt(img.rows * img.cols / 3 / max_corners_);
        cv::goodFeaturesToTrack(img, corners, max_corners_, discard_threshold_, minDist);
        if (corners.size() > 0) {
            cv::cornerSubPix(
                img, corners, cv::Size(10, 10), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, .03));
        }

        message_queue_->Enqueue(
            std::make_shared<KeypointsDetectedMessage>(event->Frame(), std::move(corners)));
    }
}

void KeypointDetectorImpl::Run() {
    std::shared_ptr<Message> message;
    while (subscription_.Dequeue(message)) {
        if (auto task = dynamic_cast<KeypointDetectorTaskMessage*>(message.get()); task) {
            tasks_.push_back(message);
        } else if (auto event = dynamic_cast<FrameLoaderEventMessage*>(message.get()); event) {
            events_.push_back(message);
        }
        int i = 0;
        for (auto it = tasks_.begin(); it != tasks_.end() && i < 3; ++i) {
            auto task = dynamic_cast<DetectKeypointsTaskMessage*>(it->get());
            if (task && task->Take()) {
                taken_frames_.insert(task->Frame());
                message_queue_->Enqueue(std::make_shared<LoadFrameTaskMessage>(task->Frame()));
                ++it;
            } else if (taken_frames_.count(task->Frame())) {
                ++it;
            } else {
                it = tasks_.erase(it);
                continue;
            }
        }
        for (auto evit = events_.begin(); evit != events_.end();) {
            if (auto event = dynamic_cast<LoadResultMessage*>(message.get());
                event && taken_frames_.count(event->Frame())) {
                ProcessEvent(event);
                evit = events_.erase(evit);
                taken_frames_.erase(event->Frame());
            } else {
                ++evit;
            }
        }
    }
}

std::shared_ptr<KeypointDetector> KeypointDetector::Create(MessageQueuePtr queue) {
    return std::make_shared<KeypointDetectorImpl>(queue);
}

KeypointDetector::~KeypointDetector() {}