#pragma once

#include <ios>
#include <sstream>
#include <string>
#include <vector>

bool ReadGyroCsv(std::istream& s, std::vector<double>& timestamps,
                 std::vector<std::tuple<double, double, double>>& rvs);