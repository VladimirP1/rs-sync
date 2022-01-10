#pragma once

#include <opencv2/core.hpp>

class TrackerImpl {
   public:
    explicit TrackerImpl(int min_corners = 100, int max_corners = 200);

    void InitCorners(const cv::Mat& img);

    std::pair<std::vector<cv::Point2f>,std::vector<cv::Point2f>>  Track(const cv::Mat& prev, const cv::Mat& cur);

    std::vector<cv::Point2f> corners_;

   private:
    double CornerQuality(const cv::Mat& gray, cv::Point2f corner);

   private:
    int min_corners_, max_corners_;
    double init_discard_tresh_{0};
};