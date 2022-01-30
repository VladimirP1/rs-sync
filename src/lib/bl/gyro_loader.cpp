#include "gyro_loader.hpp"

#include <fstream>
#include <iostream>

#include <io/bb_csv.hpp>
#include <ds/prefix_sums.hpp>
#include <ds/range_interpolated.hpp>

namespace rssync {
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

        double samplerate = SampleRate();
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
        return timestamps_.size() / (timestamps_.back() - timestamps_.front());
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