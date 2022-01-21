#include "gyro_loader.hpp"

#include <fstream>
#include <iostream>

#include <io/bb_csv.hpp>
#include <math/simple_math.hpp>
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
        std::vector<Quaternion> quaternions;

        ReadGyroCsv(in, timestamps, rvs);

        smplrate_ = timestamps.size() / (timestamps.back() - timestamps.front());

        std::cout << "Sample rate:" << smplrate_ << std::endl;

        for (auto &[x, y, z] : rvs) {
            quaternions.emplace_back(x / smplrate_, y / smplrate_, z / smplrate_);
        }

        gyro_ = GyroDsT(quaternions.begin(), quaternions.end());
    }

    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {}

    Quaternion GetRotation(double from_sec, double to_sec) const override {
        return gyro_.SoftQuery(from_sec * smplrate_, to_sec * smplrate_);
    }

   private:
    using GyroDsT = Interpolated<PrefixSums<QuaternionGroup>>;
    GyroDsT gyro_;
    double smplrate_{};
};

void RegisterGyroLoader(std::shared_ptr<IContext> ctx, std::string name, std::string filename) {
    RegisterComponent<GyroLoaderImpl>(ctx, name, filename);
}

}  // namespace rssync