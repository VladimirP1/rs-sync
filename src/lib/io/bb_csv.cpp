#include "bb_csv.hpp"

#include <sstream>
#include <string>

bool ReadGyroCsv(std::istream& s, std::vector<double>& timestamps,
                 std::vector<std::tuple<double, double, double>>& rvs) {
    std::string line;
    // Parse header
    struct {
        int ts{-1};
        int rx{-1};
        int ry{-1};
        int rz{-1};
    } fields;
    std::getline(s, line);
    {
        int idx = 0;
        std::string cell_ws;
        std::stringstream line_stream(line);
        while (std::getline(line_stream, cell_ws, ',')) {
            std::stringstream cell_stream(cell_ws);
            std::string cell;
            cell_stream >> cell;
            if (cell == "time") {
                fields.ts = idx;
            } else if (cell == "gyroADC[0]") {
                fields.rx = idx;
            } else if (cell == "gyroADC[1]") {
                fields.ry = idx;
            } else if (cell == "gyroADC[2]") {
                fields.rz = idx;
            }
            idx++;
        }
    }

    std::vector<std::pair<double, Quaternion>> ret;

    while (std::getline(s, line)) {
        std::string cell;
        std::stringstream line_stream(line);

        int idx = 0;
        double ts, rx, ry, rz;
        while (std::getline(line_stream, cell, ',')) {
            std::stringstream cell_stream(cell);
            if (idx == fields.rx) {
                cell_stream >> rx;
            } else if (idx == fields.ry) {
                cell_stream >> ry;
            } else if (idx == fields.rz) {
                cell_stream >> rz;
            } else if (idx == fields.ts) {
                cell_stream >> ts;
            }
            idx++;
        }
        // std::cout << line << std::endl;
        static constexpr double kGyroScale = M_PI / 180.;
        static constexpr double kTimeScale = 1e-6;
        ts *= kTimeScale;

        timestamps.push_back(ts);
        rvs.emplace_back(ry * kGyroScale, rz * kGyroScale, -rx * kGyroScale);
    }
    return true;
}