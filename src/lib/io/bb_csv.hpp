#pragma once

#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include <math/simple_math.hpp>

bool ReadGyroCsv(std::istream& s, std::vector<double>& timestamps,
                 std::vector<Quaternion>& quaternions);