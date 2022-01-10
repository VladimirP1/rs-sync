#include "tracking.hpp"

#include <algorithm>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

TrackerImpl::TrackerImpl(int min_corners, int max_corners)
    : min_corners_{min_corners}, max_corners_{max_corners} {}

void TrackerImpl::InitCorners(const cv::Mat& img) {
    const double discard_treshold = 1e-3;
    int minDist = std::sqrt(img.rows * img.cols / 3 / max_corners_);
    cv::goodFeaturesToTrack(img, corners_, max_corners_, discard_treshold, minDist);
    cv::cornerSubPix(
        img, corners_, cv::Size(10, 10), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20,
                         .03));

    std::vector<double> responses(corners_.size(), 0.);

    for (int i = 0; i < corners_.size(); ++i) {
        auto response = CornerQuality(img, corners_[i]);
        // std::cout << response << std::endl;
        responses[i] = response;
    }

    init_discard_tresh_ = discard_treshold *
                   *std::max_element(responses.begin(), responses.end());

    std::cout << corners_.size() << std::endl;

    auto resp_iter = responses.begin();
    auto corner_iter = corners_.begin();

    while (resp_iter != responses.end()) {
        if (*resp_iter < init_discard_tresh_) {
            resp_iter = responses.erase(resp_iter);
            corner_iter = corners_.erase(corner_iter);
        } else {
            ++resp_iter;
            ++corner_iter;
        }
    }

    std::cout << corners_.size() << std::endl;
}

std::pair<std::vector<cv::Point2f>,std::vector<cv::Point2f>> TrackerImpl::Track(const cv::Mat& prev, const cv::Mat& cur) {
    if (corners_.size() < min_corners_) {
        InitCorners(cur);
    }
    std::vector<uchar> status;
    std::vector<float> err;
    
    std::vector<cv::Point2f> old_corners, new_corners;
    cv::calcOpticalFlowPyrLK(prev, cur, corners_, new_corners, status, err, cv::Size(21, 21), 6, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, .001), 0, 1e-4);


    auto old_corner_iter = corners_.begin();
    auto new_corner_iter = new_corners.begin();
    auto status_iter = status.begin();
    std::cout << corners_.size() << std::endl;
    while (new_corner_iter != new_corners.end()) {
        // auto old_response = CornerQuality(prev, *old_corner_iter);
        auto new_response = CornerQuality(cur, *new_corner_iter);

        // std::cout << old_response << " " << new_response << std::endl;

        if (2 * new_response < init_discard_tresh_ || !*status_iter) {
            old_corner_iter = corners_.erase(old_corner_iter);
            new_corner_iter = new_corners.erase(new_corner_iter);
            status_iter = status.erase(status_iter);
        } else {
            ++old_corner_iter;
            ++new_corner_iter;
            ++status_iter;
        }
    }

    old_corners = corners_;
    corners_ = new_corners;
    
    std::cout << new_corners.size() << std::endl;
    return {old_corners, new_corners};
}

double TrackerImpl::CornerQuality(const cv::Mat& gray, cv::Point2f corner) {
    const int winsize = 7;

    cv::Mat grad_x, grad_y;
    auto roi =
        cv::Rect(static_cast<int>(corner.x) - winsize / 2,
                 static_cast<int>(corner.y) - winsize / 2, winsize, winsize);

    if (roi.x < 0 || roi.y < 0 || roi.x + winsize > gray.cols ||
        roi.y + winsize > gray.rows) {
        return 0.;
    }
    auto patch = gray(roi);

    cv::Sobel(patch, grad_x, CV_32F, 1, 0);
    cv::Sobel(patch, grad_y, CV_32F, 0, 1);

    auto m0 = cv::sum(grad_x.mul(grad_x))[0];
    auto m1 = cv::sum(grad_x.mul(grad_y))[0];
    auto m2 = cv::sum(grad_y.mul(grad_y))[0];

    const auto k = .05;
    auto response = m0 * m2 - m1 * m1 - k * (m0 + m2) * (m0 + m2);
    return std::max(response, 0.);
}