#pragma once

#include <ios>
#include <vector>

#include <Eigen/Eigen>

bool ReadGyroCsv(std::istream& s, std::vector<double>& timestamps,
                 std::vector<Eigen::Vector3d>& rvs);