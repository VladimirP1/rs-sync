
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>


#include <io/bb_csv.hpp>

#include <math/simple_math.hpp>
#include <ds/prefix_sums.hpp>
#include <ds/segment_tree.hpp>
#include <ds/range_interpolated.hpp>


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