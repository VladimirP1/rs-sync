#include "visualizer.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

#include "pair_storage.hpp"

#include <iostream>

namespace rssync {
class VisualizerImpl : public IVisualizer {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
    }

    void DimImage(cv::Mat& frame, double k) override {
        frame /= (1./k);
    }

    void OverlayMatched(cv::Mat& frame, int frame_number, bool ab, bool undistorted) override {
        PairDescription desc;
        if (!pair_storage_->Get(frame_number, desc) || (!desc.has_points && !undistorted) ||
            (!desc.has_undistorted && undistorted)) {
            return;
        }

        std::vector<long>& ids = desc.point_ids;
        std::vector<cv::Point2f>& pts =
            ab ? (undistorted ? desc.points_undistorted_a : desc.points_a)
               : (undistorted ? desc.points_undistorted_b : desc.points_b);

        std::cout << ids.size() << " " << pts.size() << std::endl;

        assert(ids.size() == pts.size());
        for (int i = 0; i < ids.size(); ++i) {
            cv::circle(frame, pts[i], 5, GetColor(ids[i]), 3);
        }
    }

    void OverlayMatchedTracks(cv::Mat& frame, int frame_number, bool undistorted) override {
        PairDescription desc;
        if (!pair_storage_->Get(frame_number, desc) || (!desc.has_points && !undistorted) ||
            (!desc.has_undistorted && undistorted)) {
            return;
        }

        std::vector<long>& ids = desc.point_ids;
        std::vector<cv::Point2f>& pts_a = undistorted ? desc.points_undistorted_a : desc.points_a;
        std::vector<cv::Point2f>& pts_b = undistorted ? desc.points_undistorted_b : desc.points_b;

        assert(ids.size() == pts.size());
        for (int i = 0; i < ids.size(); ++i) {
            cv::line(frame, pts_a[i], pts_b[i], GetColor(ids[i]), 2);
        }
    }

   private:
    cv::Scalar GetColor(long id) {
        size_t idx = static_cast<size_t>(id) % kColorPalleteSize;
        return cv::Scalar(kColorPallete[idx][2], kColorPallete[idx][1], kColorPallete[idx][0]);
    }

    static constexpr size_t kColorPalleteSize = 10;
    static constexpr uchar kColorPallete[10][3] = {
        {249, 65, 68},   {243, 114, 44}, {248, 150, 30}, {249, 132, 74}, {249, 199, 79},
        {144, 190, 109}, {67, 170, 139}, {77, 144, 142}, {87, 117, 144}, {39, 125, 161}};

   private:
    std::shared_ptr<IPairStorage> pair_storage_;
};

void RegisterVisualizer(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<VisualizerImpl>(ctx, name);
}
}  // namespace rssync