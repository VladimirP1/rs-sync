#include "frame_loader.hpp"

#include <thread>

#include <opencv2/videoio.hpp>

#include <ds/lru_cache.hpp>


namespace rssync {
class FrameLoaderImpl : public rssync::BaseComponent {
    static constexpr size_t kCacheSize = 128;

   public:
    FrameLoaderImpl(std::string filename);

    void ContextLoaded(std::weak_ptr<BaseComponent> self, std::weak_ptr<IContext> ctx) override;

    ~FrameLoaderImpl();

   private:
    void Run();
    void ProcessTask(LoadFrameTaskMessage* msg);

   private:
    int cur_frame_{-1};
    double fps_{1};
    cv::VideoCapture cap_;
    LruCache<int, std::pair<double, cv::Mat>> cache_{kCacheSize};
    std::thread worker_;
};

void FrameLoaderImpl::ContextLoaded(std::weak_ptr<BaseComponent> self,
                                    std::weak_ptr<IContext> ctx) {
    // std::cout << (*MakeMessage<LoadFrameTaskMessage>(name_, 0)).ToString() << std::endl;
    worker_ = std::thread(&FrameLoaderImpl::Run, this);
}

FrameLoaderImpl::FrameLoaderImpl(std::string filename) : cap_{filename} {}

FrameLoaderImpl::~FrameLoaderImpl() {
    Inbox().Terminate();
    worker_.join();
}

void FrameLoaderImpl::Run() {
    fps_ = cap_.get(cv::CAP_PROP_FPS);

    std::shared_ptr<Message> message;
    while (Inbox().Dequeue(message)) {
        if (auto task = dynamic_cast<LoadFrameTaskMessage*>(message.get()); task) {
            ProcessTask(task);
        }
    }
}

void FrameLoaderImpl::ProcessTask(LoadFrameTaskMessage* msg) {
    auto sctx = ctx_.lock();
    auto reply_comp = sctx ? sctx->GetComponent(msg->ReplyTo()) : std::shared_ptr<BaseComponent>();
    if (!reply_comp) {
        return;
    }
    auto desired_frame = msg->Frame();
    if (auto maybe_frame = cache_.get(desired_frame); maybe_frame) {
        auto& [ts, frame] = maybe_frame.value();
        reply_comp->Inbox().Enqueue(
            MakeMessage<LoadResultMessage>(name_, desired_frame, ts, frame));
    } else {
        cv::Mat frame;
        if (desired_frame != cur_frame_) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, desired_frame);
        }
        double cur_timestamp_ = desired_frame / fps_;
        if (cap_.read(frame)) {
            reply_comp->Inbox().Enqueue(
                MakeMessage<LoadResultMessage>(name_, desired_frame, cur_timestamp_, frame));
            cache_.put(desired_frame, {cur_timestamp_, frame});
            cur_frame_ = desired_frame + 1;
        } else {
            reply_comp->Inbox().Enqueue(MakeMessage<LoadResultMessage>(name_, -1, 0., frame));
            cur_frame_ = -1;
        }
    }
}

void RegisterFrameLoader(std::shared_ptr<IContext> ctx, std::string name, size_t max_queue, std::string filename) {
    RegisterComponentLimited<FrameLoaderImpl>(ctx, name, max_queue, filename);
}

}  // namespace rssync