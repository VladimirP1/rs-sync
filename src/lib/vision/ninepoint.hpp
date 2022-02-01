#pragma once

#include <Eigen/Core>

#include <vector>

Eigen::Matrix3d FindEssentialMat(const std::vector<Eigen::Vector3d>& points1,
                                 const std::vector<Eigen::Vector3d>& points2,
                                 std::vector<unsigned char>& mask, double threshold, int iters,
                                 double* k_out = nullptr);