#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>

class FisheyeCalibration {
   public:
    FisheyeCalibration();

    std::string Name() const;

    double Rmse() const;

    int Width() const;

    int Height() const;

    cv::Mat_<double> CameraMatrix() const;

    cv::Mat_<double> DistortionCoeffs() const;

    bool IsLoaded() const;

   private:
    std::string name_;
    double rmse_{0};
    int width_{0}, height_{0};
    cv::Mat cameraMatrix_{cv::Mat::eye(3, 3, CV_64F)};
    cv::Mat distCoeffs_{cv::Mat::zeros(4, 1, CV_64F)};
    bool isLoaded_{false};

    friend std::istream& operator>>(std::istream& s, FisheyeCalibration& calibration);
};