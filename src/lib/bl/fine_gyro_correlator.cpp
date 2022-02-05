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

    double Run(double initial_offset, double search_radius, double search_step, int start_frame,
               int end_frame) override {
        std::ofstream out{"fine_cost.csv"};
        double best_ofs{};
        double best_cost = std::numeric_limits<double>::infinity();
        for (double ofs = initial_offset - search_radius; ofs < initial_offset + search_radius;
             ofs += search_step) {
            double cost = Cost(ofs, start_frame, end_frame);
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
    double Cost(double offset, int start_frame, int end_frame) {
        double rs_coeff = .5;
        auto frame_height = calibration_provider_->GetCalibraiton().Height();

        struct PointData {
            double divider;
            bool inlier;
            GyroIntegrator::GyroThunk gyro;
            Eigen::Vector3d point_a, point_b;
        };

        double frames = 0;
        double total_time = 0;
        Eigen::Vector3d bias{0, 0, 0};
        std::vector<std::vector<PointData>> frames_data;
        std::vector<int> frame_idxs;
        pair_storage_->GetFramesWith(frame_idxs, false, true, false, false, false);
        for (int frame : frame_idxs) {
            if (frame < start_frame || frame >= end_frame) {
                continue;
            }

            PairDescription desc;
            pair_storage_->Get(frame, desc);

            if (!desc.has_undistorted) {
                continue;
            }

            frames_data.emplace_back();

            double interframe = desc.timestamp_b - desc.timestamp_a;
            std::vector<cv::Point2f> points_br(desc.points_undistorted_b.size());
            for (int i = 0; i < desc.point_ids.size(); ++i) {
                PointData d;
                double ts_a = rs_coeff * desc.points_a[i].y / frame_height,
                       ts_b = rs_coeff * desc.points_b[i].y / frame_height;
                d.gyro = integrator_.IntegrateGyro(
                    (desc.timestamp_a + ts_a * interframe + offset) * sample_rate_,
                    (desc.timestamp_b + ts_b * interframe + offset) * sample_rate_);
                auto gyro0 = d.gyro.Bias({0, 0, 0});

                Eigen::Matrix3d R =
                    Eigen::AngleAxis<double>(-gyro0.rot.norm(), gyro0.rot.normalized())
                        .toRotationMatrix();
                Eigen::Vector3d point_a, point_b;
                point_a << desc.points_undistorted_a[i].x, desc.points_undistorted_a[i].y, 1;
                point_b << desc.points_undistorted_b[i].x, desc.points_undistorted_b[i].y, 1;

                std::tie(d.point_a, d.point_b) = {point_a, point_b};

                // Rotate point from frame b
                point_b = R * point_b;
                point_b /= point_b(2, 0);

                // Now we should have only translation left
                // Rescale it assuming linear velocity
                Eigen::Vector3d u = point_b - point_a;
                d.divider = (1 + ts_b - ts_a);
                u /= d.divider;

                // Recalculate point b
                point_b = point_a + u;

                points_br[i] = cv::Point2f(point_b(0, 0), point_b(1, 0));
                frames_data.back().push_back(d);
            }

            cv::Mat_<double> R1, R2, r1, r2, t;
            std::vector<uchar> mask;

            cv::Mat E = cv::findEssentialMat(desc.points_undistorted_a, points_br, 1., {0, 0},
                                             cv::RANSAC, .99, 5e-3, 100, mask);
            // std::cout << std::accumulate(mask.begin(), mask.end(), 0) << " / " << mask.size() << std::endl;
            cv::decomposeEssentialMat(E, R1, R2, t);
            cv::Rodrigues(R1, r1);
            cv::Rodrigues(R2, r2);

            for (int i = 0; i < frames_data.back().size(); ++i)
                frames_data.back()[i].inlier = mask[i];

            if (cv::norm(r1) > cv::norm(r2)) {
                std::swap(R1, R2);
                std::swap(r1, r2);
            }

            bias += integrator_
                        .IntegrateGyro((desc.timestamp_a * interframe + offset) * sample_rate_,
                                       (desc.timestamp_b * interframe + offset) * sample_rate_)
                        .FindBias({0, 0, 0});
            frames += 1;
            total_time += interframe;
        }
        bias *= 1 / frames;
        // std::cout << bias.transpose() << std::endl;
        bias = {0,0,0};

        double cost = 0;
        for (auto& frame_data : frames_data) {
            int i = 0;
            Eigen::MatrixXd problem(frame_data.size(), 3);
            for (auto& pd : frame_data) {
                // if (!pd.inlier) continue;

                auto gyrob = pd.gyro.Bias(bias);

                Eigen::Matrix3d R =
                    Eigen::AngleAxis<double>(-gyrob.rot.norm(), gyrob.rot.normalized())
                        .toRotationMatrix();

                Eigen::Vector3d point_br = R * pd.point_b;
                point_br /= point_br(2, 0);
                point_br = (point_br - pd.point_a) / pd.divider + pd.point_a;

                problem.row(i++) = point_br.normalized().cross(pd.point_a.normalized());
            }

            Eigen::MatrixXd iproblem = problem.block(0, 0, i, 3);
            auto svd = iproblem.jacobiSvd(Eigen::ComputeFullV);
            Eigen::MatrixXd t = svd.matrixV().col(svd.matrixV().cols() - 1).normalized();


            // Reweigh the rows based on error
            Eigen::Matrix<double, Eigen::Dynamic, 1> error = iproblem * t;
            // std::cout << error << "\n";
            // error = 1. / (1. + exp(1e4 * error.array().abs()));
            error = 1. / (1. + 1e4 * error.array().abs());
            iproblem.array().colwise() *= error.array();

            // Solve again
            svd = iproblem.jacobiSvd(Eigen::ComputeFullV);
            t = svd.matrixV().col(svd.matrixV().cols() - 1).normalized();
            
            cost += (iproblem * t).norm();

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