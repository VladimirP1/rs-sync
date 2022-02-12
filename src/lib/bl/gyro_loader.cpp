#include "gyro_loader.hpp"

#include <fstream>
#include <iostream>

#include <io/bb_csv.hpp>
#include <ds/prefix_sums.hpp>
#include <ds/range_interpolated.hpp>
#include <math/rotation.h>

namespace rssync {

void ResampleGyro(std::vector<double>& timestamps, std::vector<Eigen::Vector3d>& gyro) {
    for (int i = timestamps.size(); i--;) timestamps[i] -= timestamps.front();
    double sr = timestamps.size() / timestamps.back();
    int rounded_sr = int(round(sr / 50) * 50);
    double period = 1. / rounded_sr;

    int sample = 0;
    int win_left{}, win_right{};
    double win_radius = period * 3;
    std::vector<double> new_timestamps;
    std::vector<Eigen::Vector3d> new_gyros;
    while (true) {
        double ts = sample * period;
        while (win_right < timestamps.size() && timestamps[win_right] <= ts + win_radius)
            ++win_right;
        while (win_left + 1 < win_right && timestamps[win_left + 1] < ts - win_radius) ++win_left;
        if (win_right >= timestamps.size()) break;
        Eigen::Vector3d new_gyro = Eigen::Vector3d::Zero();
        double kern_sum = 0;
        for (int i = win_left; i < win_right; ++i) {
            double k = std::max(0., 1 - std::abs((ts - timestamps[i]) / period));
            kern_sum += k;
            new_gyro += gyro[i] * k;
        }
        new_gyro /= kern_sum;
        new_gyros.push_back(new_gyro);
        new_timestamps.push_back(ts);

        // std::cout << win_left << " " << sample << " " << win_right << " " << timestamps.size()
        //           << std::endl;
        ++sample;
    }
    timestamps = new_timestamps;
    gyro = new_gyros;
}

class GyroLoaderImpl : public IGyroLoader {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {}

    GyroLoaderImpl(std::string filename) {
        if (filename == "") return;
        std::ifstream in{filename};
        if (!in) {
            throw std::runtime_error{"cannot open gyro file"};
        }

        ReadGyroCsv(in, timestamps_, gyro_);
        ResampleGyro(timestamps_, gyro_);

        double samplerate = SampleRate();

        for (auto& rv : gyro_) {
            rv /= samplerate;
        }
    }

    size_t DataSize() const override { return gyro_.size(); }

    void GetData(Eigen::Vector3d* ptr, size_t size) const override {
        std::copy_n(gyro_.begin(), std::min(size, gyro_.size()), ptr);
        auto orient = AngleAxisToQuaternion(orientation_);
        for (int i = 0; i < size; ++i) {
            ptr[i] = QuaternionToAngleAxis(orient * AngleAxisToQuaternion(ptr[i]));
        }
    }

    double SampleRate() const override {
        return (timestamps_.size() - 1) / (timestamps_.back() - timestamps_.front());
    }

    void SetOrientation(Eigen::Vector3d orientation) override { orientation_ = orientation; }

   private:
    std::vector<double> timestamps_;
    std::vector<Eigen::Vector3d> gyro_;

    Eigen::Vector3d orientation_;
};

void RegisterGyroLoader(std::shared_ptr<IContext> ctx, std::string name, std::string filename) {
    RegisterComponent<GyroLoaderImpl>(ctx, name, filename);
}

}  // namespace rssync