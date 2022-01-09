#include "segtree.hpp"

#include <math.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

class QuaternionGroup;

class Quaternion {
    static constexpr double eps = 1e-15;

    friend class QuaternionGroup;

   public:
    Quaternion() { std::fill(data_, data_ + 4, 0); }

    explicit Quaternion(double a, double b, double c, double d)
        : data_{a, b, c, d} {}

    explicit Quaternion(double rx, double ry, double rz) {
        auto norm = std::sqrt(rx * rx + ry * ry + rz * rz);
        auto a = cos(norm / 2.);
        auto k = SinxInvx(norm / 2) / 2.;
        auto b = rx * k;
        auto c = ry * k;
        auto d = rz * k;
        data_[0] = a;
        data_[1] = b;
        data_[2] = c;
        data_[3] = d;
    }

    double Norm() {
        return std::sqrt(data_[0] * data_[0] + data_[1] * data_[1] +
                         data_[2] * data_[2] + data_[3] * data_[3]);
    }

    void ToRotVec(double& rx, double& ry, double& rz) {
        auto cos = data_[0];
        auto sin_norm = std::sqrt(data_[1] * data_[1] + data_[2] * data_[2] +
                                  data_[3] * data_[3]);
        auto angle = 2 * atan2(sin_norm, cos);
        if (sin_norm < eps) {
            rx = ry = rz = 0.;
            return;
        }
        rx = data_[1] / sin_norm * angle;
        ry = data_[2] / sin_norm * angle;
        rz = data_[3] / sin_norm * angle;
    }

   private:
    double data_[4];

    double SinxInvx(double x) {
        if (std::fabs(x) < eps) {
            return 1.;
        }
        return std::sin(x) / x;
    }
};

struct QuaternionGroup {
    typedef Quaternion value_type;

    Quaternion unit() const { return Quaternion{1, 0, 0, 0}; }

    Quaternion add(Quaternion a, Quaternion b) const {
        // (a0 + b0*i + c0*j + d0*k) * (a1 + b1*i + c1*j + d1*k) =
        // (a0*a1 + a0*b1*i + a0*c1*j + a0*d1*k) (b0*a1*i + b0*b1*-1 + b0*c1*k +
        // b0*d1*-j) (c0*a1*j + c0*b1*-k + c0*c1*-1 + c0*d1*i)(d0*a1*k + d0*b1*j
        // + d0*c1*-i + d0*d1*-1) = (a0*a1 + b0*b1*-1 + c0*c1*-1  + d0*d1*-1) +
        // (a0*b1 + b0*a1 + c0*d1 - d0*c1)*i + (a0*c1 - b0*d1 + c0*a1 + d0*b1)*j
        // + (a0*d1 + b0*c1 - c0*b1 + d0*a1)*k
        double a0{a.data_[0]}, b0{a.data_[1]}, c0{a.data_[2]}, d0{a.data_[3]};
        double a1{b.data_[0]}, b1{b.data_[1]}, c1{b.data_[2]}, d1{b.data_[3]};
        return Quaternion{a0 * a1 - b0 * b1 - c0 * c1 - d0 * d1,
                          a0 * b1 + b0 * a1 + c0 * d1 - d0 * c1,
                          a0 * c1 - b0 * d1 + c0 * a1 + d0 * b1,
                          a0 * d1 + b0 * c1 - c0 * b1 + d0 * a1};
    }

    Quaternion mult(Quaternion a, double k) const {
        double x, y, z;
        a.ToRotVec(x, y, z);
        x *= k;
        y *= k;
        z *= k;
        return Quaternion(x, y, z);
    }
};

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
    // std::ifstream input("193653AA_FIXED.CSV");
    std::ifstream input("rch.csv");
    // std::ifstream input("hawk.csv");

    ReadGyroCsv(input, ts, q);

    // for (int i = 0; i < 20000; ++i) {
    //     q.push_back(Quaternion{(i % 100) / 1000.,0,0});
    // }

    SegmentTree<QuaternionGroup> t(q.begin(), q.end());
    double ofs = atoi(argv[1]) / 4000.;
    std::cerr << ofs << std::endl; 
    std::cout << "x,y,z" << std::endl;
    for (double mid = ofs; mid < 100.; mid += 1. / 30) {
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

        auto quat = t.SoftQuery(1000 * mid, 1000 * mid + 33);
        quat.ToRotVec(x, y, z);
        std::cout << x << "," << y << "," << z << std::endl;
    }

    return 0;
}