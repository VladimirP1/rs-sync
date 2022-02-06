#include "calibration_provider.hpp"

#include <mutex>
#include <fstream>

#include <vision/utils.hpp>

namespace rssync {
class CalibrationProviderImpl : public ICalibrationProvider {
   public:
    CalibrationProviderImpl(std::string filename = "") {
        if (filename == "") return;
        std::ifstream in{filename};
        if (!in) {
            throw std::runtime_error{"cannot open calibration file"};
        }
        in >> calibration_;
    }

    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {}

    virtual void SetRsCoefficent(double k) override { rs_coeff_ = k; }

    double GetRsCoefficent() const override { return rs_coeff_; }

    FisheyeCalibration GetCalibraiton() override { return calibration_; }

    cv::Mat GetReasonableProjection() override { return GetProjectionForUndistort(calibration_); }

    void SetCalibration(const FisheyeCalibration& calibration) override {
        calibration_ = calibration;
    }

   private:
    double rs_coeff_{};
    FisheyeCalibration calibration_;
    std::mutex mtx_;
};

void RegisterCalibrationProvider(std::shared_ptr<IContext> ctx, std::string name,
                                 std::string filename) {
    RegisterComponent<CalibrationProviderImpl>(ctx, name, filename);
}
}  // namespace rssync