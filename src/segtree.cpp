
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

    double d = 9e-2;
    {
        std::vector<GenericQuaternion<ceres::Jet<double, 3>>> quaternions(
            33, GenericQuaternion<ceres::Jet<double, 3>>{{1e-1, 0}, {1., 1}, {0., 2}});
        GenericQuaternionGroup<ceres::Jet<double, 3>> g;
        GenericQuaternion<ceres::Jet<double, 3>> sum = g.unit();
        for (auto& q : quaternions) {
            sum = g.add(q, sum);
        }
        std::cout << sum << std::endl;
        std::cout << sum.Bias(d, 0, 0) << std::endl;
    }

    {
        std::vector<Quaternion> quaternions(33, Quaternion{1e-1 + d, 1., 0.});
        GenericQuaternionGroup<double> g;
        GenericQuaternion<double> sum = g.unit();
        for (auto& q : quaternions) {
            sum = g.add(q, sum);
        }
        std::cout << sum << std::endl;
    }

    // QuaternionJet j(ceres::Jet<double,4>(1, 0),ceres::Jet<double,4>(1),ceres::Jet<double,4>(1));
    // QuaternionGroupJet g;
    // auto q = g.unit();
    // std::cout << q << std::endl;
    // ceres::Jet<double, 4> x,y,z;
    // q.ToRotVec(x,y,z);

    return 0;
}