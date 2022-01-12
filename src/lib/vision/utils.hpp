#pragma once

#include <opencv2/core.hpp>

#include <vision/calibration.hpp>

cv::Mat GetProjectionForUndistort(const FisheyeCalibration& calibration);

cv::Mat_<double> ProjectionTo2d(const cv::Mat_<double>& in, double z0, double w0, double z1 = 1.);
class Mosaic {
   public:
    Mosaic(const cv::Mat& output, int patch_size)
        : rows_(output.rows), cols_(output.cols), patch_size_(patch_size) {}

    bool Add(cv::Mat& out, const cv::Mat& m) {
        if (j + patch_size_ >= cols_ || i + patch_size_ >= rows_) {
            return false;
        }

        m.copyTo(out(cv::Rect(cv::Point(j, i), m.size())));

        return true;
    }

    bool Advance() {
        j += patch_size_;
        if (j + patch_size_ >= cols_) {
            j = 0;
            i += patch_size_;
        }

        if (i + patch_size_ >= rows_) {
            return false;
        }
        return true;
    }

   private:
    int i{}, j{};
    int rows_, cols_;
    int patch_size_;
};
