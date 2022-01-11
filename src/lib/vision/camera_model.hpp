#pragma once

#include <opencv2/core.hpp>

void UndistortPointJacobian(cv::Point2d uv, cv::Point2d& xy, cv::Mat& A,
                            cv::Mat camera_matrix,
                            cv::Mat distortion_cooficients);