#include "fine_gyro_correlator.hpp"

#include "pair_storage.hpp"
#include "gyro_loader.hpp"
#include "calibration_provider.hpp"

#include <math/gyro_integrator.hpp>

#include <io/stopwatch.hpp>

#include <fstream>
#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

using Eigen::Matrix;

namespace rssync {
class FineGyroCorrelatorImpl : public IFineGyroCorrelator {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);

        std::vector<Eigen::Vector3d> gyro_data(gyro_loader_->DataSize());
        gyro_loader_->GetData(gyro_data.data(), gyro_data.size());

        sample_rate_ = gyro_loader_->SampleRate();
        // LowpassGyro(gyro_data.data(), gyro_data.size(), sample_rate_ / 250.);

        integrator_ = {gyro_data.data(), static_cast<int>(gyro_data.size())};
    }

    double Run(double initial_offset, Eigen::Vector3d bias, double search_radius,
               double search_step, int start_frame, int end_frame) override {
        std::ofstream out{"fine_cost.csv"};
        double best_ofs{};
        double best_cost = std::numeric_limits<double>::infinity();
        for (double ofs = initial_offset - search_radius; ofs < initial_offset + search_radius;
             ofs += search_step) {
            double cost = Cost(ofs, bias, start_frame, end_frame);
            if (cost < best_cost) {
                best_cost = cost;
                best_ofs = ofs;
            }
            out << ofs << "," << cost << std::endl;
        }
        std::cout << "best_cost = " << best_cost << " best_ofs = " << best_ofs << std::endl;
        return best_ofs;
    }

   private:
    double Cost(double offset, Eigen::Vector3d bias, int start_frame, int end_frame) {
        double rs_coeff = calibration_provider_->GetRsCoefficent();
        auto frame_height = calibration_provider_->GetCalibraiton().Height();

        std::vector<int> frame_idxs;
        pair_storage_->GetFramesWith(frame_idxs, false, true, false, false, false);

        double cost = 0;
        for (int frame : frame_idxs) {
            if (frame < start_frame || frame >= end_frame) {
                continue;
            }

            PairDescription desc;
            pair_storage_->Get(frame, desc);

            if (!desc.has_undistorted) {
                continue;
            }

            Eigen::MatrixXd problem(desc.point_ids.size(), 3);
            double interframe = desc.timestamp_b - desc.timestamp_a;
            for (int i = 0; i < desc.point_ids.size(); ++i) {
                double ts_a = rs_coeff * desc.points_a[i].y / frame_height,
                       ts_b = rs_coeff * desc.points_b[i].y / frame_height;

                auto gyro = integrator_.IntegrateGyro(
                    (desc.timestamp_a + ts_a * interframe + offset) * sample_rate_,
                    (desc.timestamp_b + ts_b * interframe + offset) * sample_rate_);
                auto gyrob = gyro.Bias(bias);

                Eigen::Vector3d point_a, point_b;
                point_a << desc.points_undistorted_a[i].x, desc.points_undistorted_a[i].y, 1;
                point_b << desc.points_undistorted_b[i].x, desc.points_undistorted_b[i].y, 1;

                Eigen::Matrix3d R =
                    Eigen::AngleAxis<double>(-gyrob.rot.norm(), gyrob.rot.normalized())
                        .toRotationMatrix();

                Eigen::Vector3d point_br = R * point_b;
                point_br /= point_br(2, 0);
                point_br = (point_br - point_a) / (1 + ts_b - ts_a) + point_a;

                problem.row(i) = point_br.normalized().cross(point_a.normalized());
            }

            auto svd = problem.jacobiSvd(Eigen::ComputeFullV);
            auto t = svd.matrixV().col(svd.matrixV().cols() - 1).normalized().eval();

            // Reweigh the rows based on error
            auto error = (problem * t).eval();
            // std::cout << error << "\n";
            // error = 1. / (1. + exp(1e4 * error.array().abs()));
            error = 1. / (1. + 1e4 * error.array().abs());
            problem.array().colwise() *= error.array();

            // Solve again
            svd = problem.jacobiSvd(Eigen::ComputeFullV);
            t = svd.matrixV().col(svd.matrixV().cols() - 1).normalized();
            
            cost += (problem * t).norm();
        }
        return cost;
    }

    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
    std::shared_ptr<ICalibrationProvider> calibration_provider_;

    GyroIntegrator integrator_;
    double sample_rate_;
};

void RegisterFineGyroCorrelator(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<FineGyroCorrelatorImpl>(ctx, name);
}

}  // namespace rssync