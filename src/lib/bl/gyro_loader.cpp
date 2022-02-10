#include "gyro_loader.hpp"

#include <fstream>
#include <iostream>

#include <io/bb_csv.hpp>
#include <ds/prefix_sums.hpp>
#include <ds/range_interpolated.hpp>

namespace rssync {

void ResampleGyro(std::vector<double>& timestamps, std::vector<Eigen::Vector3d>& gyro) {
    double sr = timestamps.size() / (timestamps.back() - timestamps.front());
    int rounded_sr = int(round(sr / 50) * 50);
    double period = 1. / rounded_sr;
    // std::cout << rounded_sr << std::endl;

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
            double k = 1 - std::abs((ts - timestamps[i]) / period);
            if (k < 0) k = 0;
            kern_sum += k;
            new_gyro += gyro[i] * k;
        }
        new_gyro /= kern_sum;
        new_gyros.push_back(new_gyro);
        new_timestamps.push_back(ts);
        // std::cout << ts << std::endl;
        // std::cout << win_left << " " << sample << " " << win_right << " " << timestamps.size()
                //   << std::endl;
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
        // std::cout << "sr=" << samplerate << std::endl;
        for (auto& rv : gyro_) {
            rv /= samplerate;
        }
    }

    size_t DataSize() const override { return gyro_.size(); }

    void GetData(Eigen::Vector3d* ptr, size_t size) const override {
        std::copy_n(gyro_.begin(), std::min(size, gyro_.size()), ptr);
        const Eigen::AngleAxis<double> orient{orientation_.norm(), orientation_.normalized()};
        for (int i = 0; i < size; ++i) {
            const auto new_v = Eigen::AngleAxis<double>{
                orient * Eigen::AngleAxis<double>(ptr[i].norm(), ptr[i].normalized())};
            ptr[i] = new_v.axis() * new_v.angle();
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