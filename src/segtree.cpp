
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <math/simple_math.hpp>
#include <math/prefix_sums.hpp>
#include <math/segment_tree.hpp>
#include <math/range_interpolated.hpp>


bool ReadGyroCsv(std::istream& s, std::vector<double>& timestamps,
                 std::vector<Quaternion>& quaternions) {
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
        static constexpr double kGyroScale = M_PI / 180. / 1e6;
        static constexpr double kTimeScale = 1e-6;
        ts *= kTimeScale;
        Quaternion q{ry * kGyroScale, rz * kGyroScale, rx * kGyroScale};

        timestamps.push_back(ts);
        quaternions.push_back(q);

        //  std::cout << q.Norm() << std::endl;
        // std::cout << ts << " " << rx << " " << ry << " " << rz << std::endl;
    }
    return true;
}

int main(int argc, char** argv) {
    // std::vector<double> a{1,1};
    // SegmentTree<DefaultGroup<double>> t(a.begin(), a.end());

    // Quaternion q(0, 0, 1e-6);
    // double x, y, z;
    // q = QuaternionGroup().mult(q, 1);
    // q.ToRotVec(x, y, z);
    // std::cout << q.Norm() << " " << x << " " << y << " " << z << std::endl;

    std::vector<double> ts;
    std::vector<Quaternion> q;
    std::ifstream input("193653AA_FIXED.CSV");
    // std::ifstream input("rch.csv");
    // std::ifstream input("hawk.csv");

    ReadGyroCsv(input, ts, q);

    // for (int i = 0; i < 20000; ++i) {
    //     q.push_back(Quaternion{(i % 100) / 1000.,0,0});
    // }

    Interpolated<SegmentTree<QuaternionGroup>> t(q.begin(), q.end());
    std::cout << "x,y,z" << std::endl;
    for (double mid = 0; mid < 100.; mid += 1. / 1000) {
        double x, y, z;
        // double sx{}, sy{}, sz{};
        // int n = 0;
        // for (double ws = 17; ws > 2; ws /= 2) {
        //     auto quat = t.SoftQuery(1000*mid - ws, 1000*mid + ws);
        //     quat.ToRotVec(x, y, z);
        //     sx += x;
        //     sy += y;
        //     sz += z;
        //     ++n;
        // }
        // sx /= n;
        // sy /= n;
        // sz /= n;
        // std::cout << sx << "," << sy << "," << sz << std::endl;

        auto quat = t.SoftQuery(1000 * mid, 1000 * mid + 1);
        quat.ToRotVec(x, y, z);
        std::cout << x << "," << y << "," << z << std::endl;
    }

    return 0;
}