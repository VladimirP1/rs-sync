#include <iostream>
#include <stdexcept>
#include <thread>

#include <bl/frame_loader.hpp>
#include <bl/utils.hpp>
#include <ds/lru_cache.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <io/stopwatch.hpp>

namespace rssync {
class OpticalFlowLK : public rssync::BaseComponent {
    static constexpr size_t kCacheSize = 16;

   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        frame_loader_ = ctx_.lock()->GetComponent<rssync::IFrameLoader>(rssync::kFrameLoaderName);
        uuid_gen_ = ctx_.lock()->GetComponent<rssync::IUuidGen>(rssync::kUuidGenName);
    }

    bool CalcOptflow(int frame_number) {
        KeypointInfo info;
        GetKeypoints(frame_number, info);
        if (info.points.size() < min_corners_) {
            CalcKeypoints(frame_number, info);
        }
        cv::Mat prev_color, cur_color, prev, cur;
        {
            Stopwatch s("loading");
        if (!frame_loader_->GetFrame(frame_number, prev_color) ||
            !frame_loader_->GetFrame(frame_number + 1, cur_color)) {
            return false;
        }

        cv::cvtColor(prev_color, prev, cv::COLOR_BGR2GRAY);
        cv::cvtColor(cur_color, cur, cv::COLOR_BGR2GRAY);
        }

        // Track
        std::vector<uchar> status;
        std::vector<float> err;

        std::vector<cv::Point2f> new_corners;
        {
            Stopwatch s("pyrlk");
            cv::calcOpticalFlowPyrLK(
                prev, cur, info.points, new_corners, status, err, cv::Size(21, 21), 6,
                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, .001), 0,
                1e-4);
        }

        // FIlter out low quality corners
        {
            Stopwatch s("filtering");
            auto old_corner_iter = info.points.begin();
            auto new_corner_iter = new_corners.begin();
            auto ids_iter = info.ids.begin();
            auto status_iter = status.begin();
            while (new_corner_iter != new_corners.end()) {
                auto new_response = CornerQuality(cur, *new_corner_iter);

                if (2 * new_response < info.discard_threshold || !*status_iter) {
                    old_corner_iter = info.points.erase(old_corner_iter);
                    new_corner_iter = new_corners.erase(new_corner_iter);
                    status_iter = status.erase(status_iter);
                    ids_iter = info.ids.erase(ids_iter);
                } else {
                    ++old_corner_iter;
                    ++new_corner_iter;
                    ++status_iter;
                    ++ids_iter;
                }
            }
        }

        // Put into cache
        KeypointInfo new_info;
        new_info.points = new_corners;
        new_info.ids = info.ids;
        new_info.discard_threshold = info.discard_threshold;
        new_info.type = KeypointInfo::Type::kTracked;
        {
            std::unique_lock<std::mutex> lock{cache_mutex_};
            keypoint_cache_.put(frame_number + 1, new_info);
        }

        std::cout << (int)info.type << " " << info.points.size() << " " << new_corners.size()
                  << std::endl;
    }

    // private:
    struct KeypointInfo {
        enum class Type { kTracked, kCorners };
        Type type;
        double discard_threshold;
        std::vector<long> ids;
        std::vector<cv::Point2f> points;
    };

    // private:
    bool GetKeypoints(int frame_number, KeypointInfo& info) {
        {
            std::unique_lock<std::mutex> lock{cache_mutex_};
            if (auto k = keypoint_cache_.get(frame_number); k) {
                info = k.value();
                return true;
            }
        }
        return CalcKeypoints(frame_number, info);
    }

    bool CalcKeypoints(int frame_number, KeypointInfo& info) {
        cv::Mat src, img;
        if (!frame_loader_->GetFrame(frame_number, src)) {
            return false;
        }
        // Convert to gray
        cv::cvtColor(src, img, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        int minDist = std::sqrt(img.rows * img.cols / 3 / max_corners_);

        // Find corners
        cv::goodFeaturesToTrack(img, corners, max_corners_, discard_threshold_scale_, minDist);
        if (corners.size() > 0) {
            cv::cornerSubPix(
                img, corners, cv::Size(10, 10), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, .03));
        }

        // Calculate our quality metric on them
        std::vector<double> responses(corners.size(), 0.);
        for (int i = 0; i < corners.size(); ++i) {
            auto response = CornerQuality(img, corners[i]);
            responses[i] = response;
        }

        info.type = KeypointInfo::Type::kCorners;
        info.discard_threshold =
            discard_threshold_scale_ * (*std::max_element(responses.begin(), responses.end()));
        std::generate_n(std::inserter(info.ids, info.ids.end()), corners.size(),
                        [this]() { return uuid_gen_->Next(); });
        info.points = std::move(corners);
        {
            std::unique_lock<std::mutex> lock{cache_mutex_};
            keypoint_cache_.put(frame_number, info);
        }
        return true;
    }

    double CornerQuality(const cv::Mat& gray, cv::Point2f corner) {
        static constexpr int winsize = 7;

        cv::Mat grad_x, grad_y;
        auto roi = cv::Rect(static_cast<int>(corner.x) - winsize / 2,
                            static_cast<int>(corner.y) - winsize / 2, winsize, winsize);

        if (roi.x < 0 || roi.y < 0 || roi.x + winsize > gray.cols || roi.y + winsize > gray.rows) {
            return 0.;
        }
        auto patch = gray(roi);

        cv::Sobel(patch, grad_x, CV_32F, 1, 0);
        cv::Sobel(patch, grad_y, CV_32F, 0, 1);

        auto m0 = cv::sum(grad_x.mul(grad_x))[0];
        auto m1 = cv::sum(grad_x.mul(grad_y))[0];
        auto m2 = cv::sum(grad_y.mul(grad_y))[0];

        static constexpr double k = .05;
        auto response = m0 * m2 - m1 * m1 - k * (m0 + m2) * (m0 + m2);
        return std::max(response, 0.);
    }

   private:
    std::shared_ptr<IFrameLoader> frame_loader_;
    std::shared_ptr<IUuidGen> uuid_gen_;
    LruCache<int, KeypointInfo> keypoint_cache_{kCacheSize};
    std::mutex cache_mutex_;

    int min_corners_{70};
    int max_corners_{700};
    double discard_threshold_scale_{1e-3};
};
}  // namespace rssync

int main() {
    auto ctx = rssync::IContext::CreateContext();

    rssync::RegisterFrameLoader(ctx, rssync::kFrameLoaderName, "141101AA.MP4");

    rssync::RegisterUuidGen(ctx, rssync::kUuidGenName);

    // rssync::RegisterKeypointDetector(ctx, rssync::kKeypointDetectorName);

    rssync::RegisterComponent<rssync::OpticalFlowLK>(ctx, "OpticalFlowLK");

    ctx->ContextLoaded();

    for (int i = 0; i < 800; ++i) {
        // cv::Mat out;
        // ctx->GetComponent<rssync::IFrameLoader>(rssync::kFrameLoaderName)->GetFrame(i, out);
        // std::cout << out.cols << std::endl;
        // rssync::OpticalFlowLK::KeypointInfo info;
        // ctx->GetComponent<rssync::OpticalFlowLK>("OpticalFlowLK")->GetKeypoints(i, info);
        ctx->GetComponent<rssync::OpticalFlowLK>("OpticalFlowLK")->CalcOptflow(i);
        // std::cout << info.points.size() << std::endl;
    }

    std::cout << "main done" << std::endl;

    return 0;
}