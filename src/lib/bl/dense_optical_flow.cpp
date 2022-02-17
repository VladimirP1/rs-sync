#include "optical_flow.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow.hpp>

#include "frame_loader.hpp"
#include "pair_storage.hpp"
#include "utils.hpp"

namespace rssync {
class OpticalFlowDense : public IOpticalFlow {
    static constexpr size_t kCacheSize = 16;

   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        frame_loader_ = ctx_.lock()->GetComponent<IFrameLoader>(kFrameLoaderName);
        uuid_gen_ = ctx_.lock()->GetComponent<IUuidGen>(rssync::kUuidGenName);
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
    }

    bool CalcOptflow(int frame_number) override {
        double timestamp_a, timestamp_b;
        cv::Mat prev_color, cur_color, prev, cur, flow;
        if (!frame_loader_->GetFrame(frame_number, prev_color, &timestamp_a) ||
            !frame_loader_->GetFrame(frame_number + 1, cur_color, &timestamp_b)) {
            return false;
        }

        cv::cvtColor(prev_color, prev, cv::COLOR_BGR2GRAY);
        cv::cvtColor(cur_color, cur, cv::COLOR_BGR2GRAY);

        of_->calc(prev, cur, flow);

        PairDescription desc;
        desc.timestamp_a = timestamp_a;
        desc.timestamp_b = timestamp_b;
        desc.point_ids.clear();
        desc.points_a.clear();
        desc.points_b.clear();
        desc.has_points = true;

        static constexpr int step = 200;
        for (int i = step; i < prev.cols; i += step) {
            for (int j = step; j < prev.rows; j += step) {
                desc.point_ids.push_back(i * prev.rows + j);
                desc.points_a.push_back(cv::Point2f(i, j));
                desc.points_b.push_back(cv::Point2f(i + flow.at<cv::Vec2f>(j, i)[0], j + flow.at<cv::Vec2f>(j, i)[1]));
            }
        }

        // Put it into storage
        pair_storage_->Update(frame_number, desc);

        return true;
    }

   private:
    std::shared_ptr<IFrameLoader> frame_loader_;
    std::shared_ptr<IUuidGen> uuid_gen_;
    std::shared_ptr<IPairStorage> pair_storage_;

    cv::Ptr<cv::DISOpticalFlow> of_ = cv::DISOpticalFlow::create();
};

void RegisterOpticalFlowDense(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<OpticalFlowDense>(ctx, name);
}
}  // namespace rssync