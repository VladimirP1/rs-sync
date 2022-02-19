#include "fine_sync.hpp"

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
#include <random>

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

using Eigen::Matrix;

namespace rssync {

class FineSyncImpl : public IFineSync {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);

        std::vector<Eigen::Vector3d> gyro_data(gyro_loader_->DataSize());
        gyro_loader_->GetData(gyro_data.data(), gyro_data.size());

        sample_rate_ = gyro_loader_->SampleRate();
        integrator_ = {gyro_data.data(), static_cast<int>(gyro_data.size())};
        LowpassGyro(gyro_data.data(), gyro_data.size(), sample_rate_ / 50.);
        integrator_lpf_ = {gyro_data.data(), static_cast<int>(gyro_data.size())};
    }

    double Run2(double initial_offset, Eigen::Vector3d bias, int start_frame,
                int end_frame) override {
        std::ofstream out("sync2.csv");

        for (double ofs = initial_offset - 2e-3; ofs < initial_offset + 2e-3; ofs += 1e-5) {
            double cost = 0;
            for (int frame = start_frame; frame < end_frame; ++frame) {
                PairDescription desc;
                if (!pair_storage_->Get(frame, desc) || !desc.has_undistorted) continue;
                cost += FrameCost(frame, ofs, {0, 0, 0}, {0, 0, 0});
            }
            out << ofs * 1000 << "," << cost << std::endl;
        }

        return initial_offset;
    }

    double Run(double initial_offset, Eigen::Vector3d bias, int start_frame,
               int end_frame) override {
        optim::algo_settings_t settings;
        settings.de_settings.n_pop = 20;
        settings.de_settings.n_gen = 100;

        initial_offset *= 1000;
        bias *= sample_rate_;
        Eigen::VectorXd init_sol(4, 1);
        init_sol << initial_offset, bias(0, 0), bias(1, 0), bias(2, 0);

        settings.vals_bound = true;
        settings.lower_bounds.resize(4, 1);
        settings.upper_bounds.resize(4, 1);
        settings.lower_bounds << init_sol(0, 0) - 15, -.1, -.1, -.1;
        settings.upper_bounds << init_sol(0, 0) + 15, +.1, +.1, +.1;
        
        optim::de(
            init_sol,
            [&](const Eigen::VectorXd& params, Eigen::VectorXd*, void*) {
                double cost = 0;
                for (int frame = start_frame; frame < end_frame; ++frame) {
                    PairDescription desc;
                    if (!pair_storage_->Get(frame, desc) || !desc.has_undistorted) continue;
                    auto bias =
                        Eigen::Vector3d{params(1, 0), params(2, 0), params(3, 0)} / sample_rate_;
                    cost += FrameCost(frame, params(0, 0) / 1000., bias, {0, 0, 0});
                }
                return cost;
            },
            nullptr, settings);
        std ::cout << "de-sync: " << init_sol(0, 0) << " | " << init_sol.block(1, 0, 3, 1)
                   << std::endl;
        return init_sol(0, 0) / 1000.;
    }

    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
    std::shared_ptr<ICalibrationProvider> calibration_provider_;

    GyroIntegrator integrator_;
    GyroIntegrator integrator_lpf_;
    double sample_rate_;

   private:
    double FrameCost(int frame, double gyro_delay, Eigen::Vector3d bias, Eigen::Vector3d align,
                     double hf_attenuate = 1) {
        PairDescription desc_;
        pair_storage_->Get(frame, desc_);

        double rs_coeff = calibration_provider_->GetRsCoefficent();
        auto frame_height = calibration_provider_->GetCalibraiton().Height();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> problem(desc_.point_ids.size(), 3);
        double interframe = desc_.timestamp_b - desc_.timestamp_a;
        for (int i = 0; i < desc_.point_ids.size(); ++i) {
            double ts_a = rs_coeff * desc_.points_a[i].y / frame_height,
                   ts_b = rs_coeff * desc_.points_b[i].y / frame_height;

            auto gyro =
                integrator_lpf_
                    .IntegrateGyro(
                        (desc_.timestamp_a + ts_a * interframe + gyro_delay) * sample_rate_,
                        (desc_.timestamp_b + ts_b * interframe + gyro_delay) * sample_rate_)
                    .Mix(integrator_.IntegrateGyro(
                             (desc_.timestamp_a + ts_a * interframe + gyro_delay) * sample_rate_,
                             (desc_.timestamp_b + ts_b * interframe + gyro_delay) * sample_rate_),
                         hf_attenuate);

            auto gyrob = gyro.Bias(bias);

            Eigen::Vector3d point_a, point_b;
            point_a << desc_.points_undistorted_a[i].x, desc_.points_undistorted_a[i].y, 1;
            point_b << desc_.points_undistorted_b[i].x, desc_.points_undistorted_b[i].y, 1;

            Eigen::Matrix3d R_align = AngleAxisToRotationMatrix(align);
            Eigen::Matrix3d R = AngleAxisToRotationMatrix((-gyrob.rot).eval());
            Eigen::Vector3d point_br = (R_align * R * R_align.transpose()) * point_b;
            point_br /= point_br(2, 0);
            point_br = (point_br - point_a) / (1 + ts_b - ts_a) + point_a;

            problem.row(i) = point_br.normalized().cross(point_a.normalized());
            problem.row(i).normalize();
        }

        // std::ofstream out("problem.xyz");
        // for (int i = 0; i < problem.rows(); ++i) {
        //     out << std::fixed << std::setprecision(16) << problem.row(i) << std::endl;
        // }
        // exit(0);

        Eigen::Matrix<double, -1, 1> weights(problem.rows(), 1);
        weights.setOnes();
        { /* https://hal.inria.fr/inria-00074015/document */
            Eigen::Vector3d best_sol;
            double least_med = std::numeric_limits<double>::infinity();
            std::mt19937 gen;
            std::normal_distribution<double> normal;
            for (int i = 0; i < 20; ++i) {
                auto v = Eigen::Vector3d{normal(gen), normal(gen), normal(gen)}.normalized().eval();
                auto residuals = (problem * v).array().eval();
                auto residuals2 = (residuals * residuals).eval();

                std::sort(residuals2.data(), residuals2.data() + residuals2.rows());
                auto med = residuals2(residuals2.rows() / 4, 0);
                if (med < least_med) {
                    least_med = med;
                    best_sol = v;
                }
            }
            auto residuals = (problem * best_sol).array().eval();
            auto k = 1e1;
            weights = (1 / (1 + residuals * residuals * k * k)).sqrt().matrix();
            for (int i = 0; i < 10; ++i) {
                auto svd = (problem.array().colwise() * weights.array())
                               .matrix()
                               .jacobiSvd(Eigen::ComputeFullV);
                auto V = svd.matrixV().eval();
                residuals = (problem * V.col(V.cols() - 1)).array().eval();
                weights = (1 / (1 + residuals * residuals * k * k)).sqrt().matrix();
            }

            problem = problem.array().colwise() * weights.array();
        }

        auto svd = problem.jacobiSvd();

        auto S = svd.singularValues().eval();

        return S(S.rows() - 1, 0);
    }
};

void RegisterFineSync(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<FineSyncImpl>(ctx, name);
}

}  // namespace rssync