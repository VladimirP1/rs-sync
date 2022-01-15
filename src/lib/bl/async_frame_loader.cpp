#include "async_frame_loader.hpp"

#include <set>
#include <mutex>
#include <condition_variable>

#include <opencv2/videoio.hpp>

struct Task {
    Task(cv::VideoCapture* cap) : cap_(cap) {}
    virtual ~Task() {}
    virtual void Run() = 0;

   protected:
    cv::VideoCapture* cap_;
};

struct LoadIntoContextTask : public Task {
    LoadIntoContextTask(std::shared_ptr<SyncContext> ctx, cv::VideoCapture* cap, int begin, int end)
        : Task(cap), begin_(begin), end_(end), ctx_(ctx) {}

    std::future<void> GetFuture() { return promise_.get_future(); }

    void Run() override {
        cv::Mat frame;
        cap_->set(cv::CAP_PROP_POS_FRAMES, begin_);
        for (int i = begin_; i < end_; ++i) {
            cap_->read(frame);
            ctx_->InFrameContext<void>(
                i, [](FrameContext& ctx, cv::Mat& frame) { ctx.SetCachedFrame(frame.clone()); },
                frame);
        }
    }

   private:
    int begin_, end_;
    std::shared_ptr<SyncContext> ctx_;
    std::promise<void> promise_;
};

struct GetFrameCountTask : public Task {
    GetFrameCountTask(cv::VideoCapture* cap) : Task(cap) {}

    std::future<int> GetFuture() { return promise_.get_future(); }

    void Run() override { promise_.set_value(cap_->get(cv::CAP_PROP_FRAME_COUNT)); }

   private:
    std::promise<int> promise_;
};

struct GetFrameNumberAtTask : public Task {
    GetFrameNumberAtTask(cv::VideoCapture* cap, double msec) : Task(cap), msec_(msec) {}

    std::future<int> GetFuture() { return promise_.get_future(); }

    void Run() override {
        cap_->set(cv::CAP_PROP_POS_MSEC, msec_);
        promise_.set_value(cap_->get(cv::CAP_PROP_FRAME_COUNT));
    }

   private:
    double msec_;
    std::promise<int> promise_;
};

class AsyncFrameLoaderImpl : public AsyncFrameLoader {
   public:
    AsyncFrameLoaderImpl(std::string s);
    ~AsyncFrameLoaderImpl() override {
        if (worker_.joinable()) {
            worker_.join();
        }
    }
    std::future<int> GetFrameNumberAt(double timestamp) override;
    std::future<int> GetFrameCount() override;
    std::future<void> LoadFramesIntoContext(int begin, int end,
                                            std::shared_ptr<SyncContext> ctx) override;

   private:
    void Work();
    void AppendQueue(std::unique_ptr<Task> t);

    struct TaskCompare {
        bool operator()(const std::unique_ptr<Task>& a, const std::unique_ptr<Task>& b) const {
            return a.get() < b.get();
        }
    };
    std::set<std::unique_ptr<Task>, TaskCompare> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool terminate_;
    std::thread worker_;
    cv::VideoCapture cap_;
};

void AsyncFrameLoaderImpl::AppendQueue(std::unique_ptr<Task> t) {
    std::unique_lock<std::mutex> lock{queue_mutex_};
    bool have_to_launch = queue_.empty();
    queue_.insert(std::move(t));
    queue_cv_.notify_all();
}

void AsyncFrameLoaderImpl::Work() {
    while (true) {
        std::unique_ptr<Task> t;
        {
            std::unique_lock<std::mutex> lock{queue_mutex_};
            queue_cv_.wait(lock, [this]() { return !queue_.empty() ||; });
            t = std::move(queue_.extract(queue_.begin()).value());
        }
        t->Run();
    }
}

AsyncFrameLoaderImpl::AsyncFrameLoaderImpl(std::string s) : cap_{s} {
    if (!cap_.isOpened()) {
        throw std::runtime_error{"Cannot open video file"};
    }
    worker_ = std::thread(&AsyncFrameLoaderImpl::Work, this);
}

std::future<int> AsyncFrameLoaderImpl::GetFrameNumberAt(double timestamp) {
    auto ptr = std::make_unique<GetFrameNumberAtTask>(&cap_, timestamp);
    auto future = ptr->GetFuture();
    AppendQueue(std::move(ptr));
    return future;
}
std::future<int> AsyncFrameLoaderImpl::GetFrameCount() {
    auto ptr = std::make_unique<GetFrameCountTask>(&cap_);
    auto future = ptr->GetFuture();
    AppendQueue(std::move(ptr));
    return future;
}
std::future<void> AsyncFrameLoaderImpl::LoadFramesIntoContext(int begin, int end,
                                                              std::shared_ptr<SyncContext> ctx) {
    auto ptr = std::make_unique<LoadIntoContextTask>(ctx, &cap_, begin, end);
    auto future = ptr->GetFuture();
    AppendQueue(std::move(ptr));
    return future;
}

AsyncFrameLoader::~AsyncFrameLoader() {}

std::shared_ptr<AsyncFrameLoader> AsyncFrameLoader::CreateFrameLoader(std::string path) {
    return std::make_shared<AsyncFrameLoaderImpl>(path);
}
