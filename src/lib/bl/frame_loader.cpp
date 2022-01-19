#include "frame_loader.hpp"

#include <thread>

#include <opencv2/videoio.hpp>

#include <ds/lru_cache.hpp>

namespace rssync {
class FrameLoaderImpl : public IFrameLoader {
    static constexpr size_t kCacheSize = 3;

   public:
    FrameLoaderImpl(std::string filename);

    void ContextLoaded(std::weak_ptr<BaseComponent> self) override;
    bool GetFrame(int n, cv::Mat& out) override;

    ~FrameLoaderImpl();

   private:
   private:
    int cur_frame_{-1};
    double fps_{1};
    cv::VideoCapture cap_;
    LruCache<int, std::pair<double, cv::Mat>> cache_{kCacheSize};
    std::mutex mtx_;
};

void FrameLoaderImpl::ContextLoaded(std::weak_ptr<BaseComponent> self) {}

FrameLoaderImpl::FrameLoaderImpl(std::string filename) : cap_{filename} {
    fps_ = cap_.get(cv::CAP_PROP_FPS);
}

FrameLoaderImpl::~FrameLoaderImpl() {}

bool FrameLoaderImpl::GetFrame(int frame_number, cv::Mat& out) {
    std::unique_lock<std::mutex> lock{mtx_};
    if (auto maybe_frame = cache_.get(frame_number); maybe_frame) {
        auto& [ts, frame] = maybe_frame.value();
        out = frame;
        return true;
    }

    cv::Mat frame;
    if (frame_number != cur_frame_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, frame_number);
    }
    double cur_timestamp_ = frame_number / fps_;
    if (cap_.read(frame)) {
        cache_.put(frame_number, {cur_timestamp_, frame});
        cur_frame_ = frame_number + 1;
        out = frame;
        return true;
    }
    cur_frame_ = -1;

    return false;
}

void RegisterFrameLoader(std::shared_ptr<IContext> ctx, std::string name, std::string filename) {
    RegisterComponent<FrameLoaderImpl>(ctx, name, filename);
}

}  // namespace rssync