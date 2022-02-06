#pragma once

#include "component.hpp"

#include <vector>
#include <functional>

#include <opencv2/core.hpp>

#include "normal_fitter.hpp"

namespace rssync {
void RegisterPairStorage(std::shared_ptr<IContext> ctx, std::string name);

struct PairDescription {
    double timestamp_a{}, timestamp_b{};
    std::vector<long> point_ids;
    std::vector<cv::Point2f> points_a, points_b;
    std::vector<cv::Point2f> points_undistorted_a, points_undistorted_b;
    std::vector<uchar> mask_essential, mask_4d, mask_correlation;
    cv::Mat_<double> R, t, points4d;

    std::vector<std::pair<cv::Mat, cv::Mat>> patch_transforms;

    std::vector<cv::Mat> debug_correlations;
    std::vector<std::pair<cv::Mat, cv::Mat>> debug_patches;

    bool enable_debug{};
    bool has_points{}, has_undistorted{}, has_pose{}, has_points4d{}, has_correlations{};
};

class IPairStorage : public rssync::BaseComponent {
   public:
    virtual void Update(int frame, const PairDescription& desc) = 0;
    virtual bool Get(int frame, PairDescription& desc) = 0;
    virtual bool Drop(int frame) = 0;
    virtual void GetFramesWith(std::vector<int>& out, bool points, bool undistorted, bool pose,
                               bool points4d, bool correlations) = 0;
};

constexpr const char* kPairStorageName = "PairStorage";

}  // namespace rssync