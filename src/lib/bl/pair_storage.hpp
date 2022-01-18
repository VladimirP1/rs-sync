#pragma once

#include "component.hpp"

#include <vector>
#include <functional>

#include <opencv2/core.hpp>

namespace rssync {
void RegisterPairStorage(std::shared_ptr<IContext> ctx, std::string name);

struct PairDescription {
    std::vector<long> point_ids;
    std::vector<cv::Point2f> points_a, points_b;
    std::vector<cv::Point2f> points_undistorted_a, points_undistorted_b;
    std::vector<uchar> mask_essential, mask_4d;
    cv::Mat_<double> R, t, points4d;

    std::vector<cv::Mat> _debug_0_, _debug_1_, _debug_2_, _debug_3_;

    bool has_points{}, has_undistorted{}, has_pose{}, has_points4d{};
};

class IPairStorage : public rssync::BaseComponent {
   public:
    virtual void Update(int frame, const PairDescription& desc) = 0;
    virtual bool Get(int frame, PairDescription& desc) = 0;
    virtual bool Drop(int frame) = 0;
    virtual void GetFramesWith(std::vector<int> out, bool points, bool undistorted, bool pose,
                               bool points4d) = 0;
};

constexpr const char* kPairStorageName = "PairStorage";

}  // namespace rssync