#pragma once

#include <opencv2/core.hpp>

void UndistortPointJacobian(cv::Point2d uv, cv::Point2d& xy, cv::Mat& A, cv::Mat camera_matrix,
                            cv::Mat distortion_cooficients);

void ProjectPointJacobian(cv::Point3d xyz, cv::Point2d& uv, cv::Mat& A, cv::Mat camera_matrix,
                          cv::Mat distortion_cooficients);

void ProjectPointJacobianExtended(double const* xyz, double const* lens_model, double* uv,
                                  double* du_out, double* dv_out);