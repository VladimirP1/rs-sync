#include "keypoint_detector.hpp"
#include "frame_loader.hpp"
#include "component.hpp"


#include <iostream>
#include <vector>
#include <thread>

#include <opencv2/video/tracking.hpp>

namespace rssync {
class KeypointDetectorImpl : public BaseComponent {
   public:
    KeypointDetectorImpl();

    void Run();

    void Worker();

    ~KeypointDetectorImpl();

   private:
    void ContextLoaded(std::weak_ptr<BaseComponent> self, std::weak_ptr<IContext> ctx) override;

    void ProcessEvent(std::string name, int frame, cv::Mat img);

   private:
    std::vector<std::thread> threads_;

    BlockingQueue<std::pair<std::string, int>> work_;
    std::shared_ptr<BlockingMulticastQueue<std::pair<int, cv::Mat>>> imgs_ =
        BlockingMulticastQueue<std::pair<int, cv::Mat>>::Create();

    int min_corners_{70};
    int max_corners_{700};
    double discard_threshold_{1e-3};
};

KeypointDetectorImpl::KeypointDetectorImpl() {}

void KeypointDetectorImpl::ContextLoaded(std::weak_ptr<BaseComponent> self,
                                         std::weak_ptr<IContext> ctx) {
    for (int i = 0; i < 32; ++i) {
        threads_.emplace_back(&KeypointDetectorImpl::Worker, this);
    }
    threads_.emplace_back(&KeypointDetectorImpl::Run, this);
}

void KeypointDetectorImpl::Run() {
    std::shared_ptr<Message> msg;
    while (Inbox().Dequeue(msg)) {
        auto sctx = ctx_.lock();
        auto loader_comp =
            sctx ? sctx->GetComponent(rssync::kFrameLoaderName) : std::shared_ptr<BaseComponent>{};
        if (!loader_comp) {
            return;
        }
        if (auto task = dynamic_cast<DetectKeypointsTaskMessage*>(msg.get()); task) {
            work_.Enqueue({task->ReplyTo(), task->Frame()});
            loader_comp->Inbox().Enqueue(MakeMessage<LoadFrameTaskMessage>(name_, task->Frame()));
        } else if (auto event = dynamic_cast<LoadResultMessage*>(msg.get()); event) {
            imgs_->Enqueue({event->Frame(), event->Image()});
        }
    }
}

void KeypointDetectorImpl::Worker() {
    std::pair<std::string, int> my_frame;
    auto sub = imgs_->Subscribe();
    while (work_.Dequeue(my_frame)) {
        std::pair<int, cv::Mat> data;
        while (sub.Dequeue(data)) {
            if (data.first == my_frame.second) break;
        }
        std::cout << "processing" << data.first << std::endl;
        ProcessEvent(my_frame.first, my_frame.second, data.second);
                std::cout << "ok" << data.first << std::endl;

    }
}
void KeypointDetectorImpl::ProcessEvent(std::string reply_name, int frame, cv::Mat src) {
    auto sctx = ctx_.lock();
    auto reply_comp = sctx ? sctx->GetComponent(reply_name) : std::shared_ptr<BaseComponent>{};
    if (!reply_comp) {
        return;
    }

    cv::Mat img;
    cv::cvtColor(src, img, cv::COLOR_BGR2GRAY);
    // std::cout << img.cols << std::endl;

    std::vector<cv::Point2f> corners;
    int minDist = std::sqrt(img.rows * img.cols / 3 / max_corners_);
    cv::goodFeaturesToTrack(img, corners, max_corners_, discard_threshold_, minDist);
    if (corners.size() > 0) {
        cv::cornerSubPix(
            img, corners, cv::Size(10, 10), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, .03));
    }

    reply_comp->Inbox().Enqueue(
        std::make_shared<KeypointsDetectedMessage>(frame, std::move(corners)));
}

KeypointDetectorImpl::~KeypointDetectorImpl() {
    Inbox().Terminate();
    work_.Terminate();
    imgs_->Terminate();
    for (auto& t : threads_) {
        t.join();
    }
}

std::shared_ptr<BaseComponent> RegisterKeypointDetector(std::shared_ptr<IContext> ctx,
                                                        std::string name, size_t max_queue) {
    return RegisterComponentLimited<KeypointDetectorImpl>(ctx, name, max_queue);
}

}  // namespace rssync