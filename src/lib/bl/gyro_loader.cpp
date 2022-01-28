#include "gyro_loader.hpp"

#include <fstream>
#include <iostream>

#include <io/bb_csv.hpp>
#include <math/quaternion.hpp>
#include <ds/prefix_sums.hpp>
#include <ds/range_interpolated.hpp>

namespace rssync {
class GyroLoaderImpl : public IGyroLoader {
    static constexpr size_t kCacheSize = 3;

   public:
    GyroLoaderImpl(std::string filename) {
        if (filename == "") return;
        std::ifstream in{filename};
        if (!in) {
            throw std::runtime_error{"cannot open gyro file"};
        }
        // TODO: use timestamps
        std::vector<double> timestamps;
        std::vector<std::tuple<double, double, double>> rvs;
        std::vector<QuatT> quaternions;

        ReadGyroCsv(in, timestamps, rvs);

        smplrate_ = timestamps.size() / (timestamps.back() - timestamps.front());

        std::cout << "Sample rate:" << smplrate_ << std::endl;

        for (auto& [x, y, z] : rvs) {
            quaternions.push_back(QuatT::FromRotationVector(
                {{x / smplrate_, 0}, {y / smplrate_, 1}, {z / smplrate_, 2}}));
            raw_gyro_.push_back(Quaternion<double>::FromRotationVector(
                {x / smplrate_, y / smplrate_, z / smplrate_}));
        }

        gyro_ = GyroDsT(quaternions.begin(), quaternions.end());
    }

    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {}

    QuatT GetRotation(double from_sec, double to_sec) const override {
        return orientation_ * gyro_.SoftQuery(from_sec * smplrate_, to_sec * smplrate_);
    }

    bool GetRawRvs(int radius, double center, double& actual_start,double& act_step, 
                           Eigen::Vector3d* out_rvs) const override {
        int index = std::round(center * smplrate_ - .5);
        if (index + radius >= raw_gyro_.size() || index - radius < 0) {
            return false;
        }
        actual_start = (index - radius + .5) / smplrate_;
        act_step = 1./smplrate_;

        for (int i = 0; i < 2*radius + 1; ++i) {
            out_rvs[i] = (Bias(orientation_,{}) * raw_gyro_[index + i - radius]).ToRotationVector();
        }
        return true;
    }

    void SetOrientation(Quaternion<double> orient) override {
        orientation_ = {Jet<double, 3>{orient.w()}, Jet<double, 3>{orient.x()},
                        Jet<double, 3>{orient.y()}, Jet<double, 3>{orient.z()}};
    }

   private:
    using QuatGroupT = QuaternionGroup<QuatT>;
    using GyroDsT = Interpolated<PrefixSums<QuatGroupT>>;
    GyroDsT gyro_;
    std::vector<Quaternion<double>> raw_gyro_;

    double smplrate_{};
    Quaternion<Jet<double, 3>> orientation_ = Quaternion<Jet<double, 3>>::Identity();
};

void RegisterGyroLoader(std::shared_ptr<IContext> ctx, std::string name, std::string filename) {
    RegisterComponent<GyroLoaderImpl>(ctx, name, filename);
}

}  // namespace rssync