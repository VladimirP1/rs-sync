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

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

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
        LowpassGyro(gyro_data.data(), gyro_data.size(), sample_rate_ / 100.);
        integrator_lpf_ = {gyro_data.data(), static_cast<int>(gyro_data.size())};
    }

    double Run(double initial_offset, Eigen::Vector3d bias, double search_radius,
               double search_step, int start_frame, int end_frame) override {
        std::ofstream out{"fine_cost0.csv"};

        static constexpr int hist_len = 20;
        double ofs_hist[hist_len];
        int hist_pos = 0;

        int cosec_bias_upds = 0;

        Eigen::Vector3d m_bias, v_bias, g_bias;
        v_bias.setZero();
        m_bias.setZero();
        double ou = 0;
        double ofs = initial_offset, cost{};
        double m{}, v{.01}, g{};
        double b1{.85}, b2{.9}, eps{1e-8}, eta{5e-4};
        double b1_bias{.85}, b2_bias{.9}, eta_bias{1e-5};
        for (int i = 0; i < 500; ++i) {
            cost = Cost(ofs, bias, start_frame, end_frame, ou, g, g_bias);
            m = b1 * m + (1 - b1) * g;
            v = b2 * v + (1 - b2) * (g * g);
            double m_ = m / (1 - b1);
            double v_ = v / (1 - b2);
            ofs = ofs - eta / (sqrt(v_) + eps) * m_;

            m_bias = b1_bias * m_bias + (1 - b1_bias) * g_bias;
            v_bias = b1_bias * v_bias + (1 - b1_bias) * (g_bias.array() * g_bias.array()).matrix();
            Eigen::Vector3d m__bias = m_bias / (1 - b1_bias);
            Eigen::Vector3d v__bias = v_bias / (1 - b2_bias);

            ofs_hist[hist_pos] = ofs;

            auto [min_ofs_hist, max_ofs_hist] = std::minmax_element(ofs_hist, ofs_hist + hist_len);
            if (*max_ofs_hist - *min_ofs_hist < 5e-3) {
                bias = bias - (eta_bias / (sqrt(v__bias.array()) + eps) * m__bias.array()).matrix();
                std::cout << "update " << v__bias.norm() << std::endl;
                ++cosec_bias_upds;
            } else {
                cosec_bias_upds = 0;
            }

            // Termination
            if (*max_ofs_hist - *min_ofs_hist < 5e-4 && ou < 1) {
                ou = std::min(ou + .01, 1.);
            } else if (cosec_bias_upds > 30 && i > hist_len && *max_ofs_hist - *min_ofs_hist < 5e-4) {
                std::cout << "Converged in " << i << " iterations" << std::endl;
                break;
            }
            hist_pos = (hist_pos + 1) % hist_len;

            std::cout << "g=" << g << " v_=" << v_ << " ofs=" << ofs << " g_bias=" << g_bias(0, 0)
                      << " " << g_bias(1, 0) << " " << g_bias(2, 0) << std::endl;
            out << i << "," << g << "," << ofs << "," << v_ << "," << m_ << "," << bias.norm()
                << "," << cost << std::endl;
        }

        return ofs;
    }

    double Run2(double initial_offset, Eigen::Vector3d bias, double search_radius,
                double search_step, int start_frame, int end_frame) override {
        std::ofstream out{"fine_cost.csv"};
        double best_ofs{};
        double best_cost = std::numeric_limits<double>::infinity();
        for (double ofs = initial_offset - search_radius; ofs < initial_offset + search_radius;
             ofs += search_step) {
            double der;
            Eigen::Vector3d d_bias;
            double cost = Cost(ofs, bias, start_frame, end_frame, 1, der, d_bias);
            if (cost < best_cost) {
                best_cost = cost;
                best_ofs = ofs;
            }
            out << std::fixed << std::setprecision(16) << ofs << "," << cost << "," << der
                << std::endl;
        }
        std::cout << "best_cost = " << best_cost << " best_ofs = " << best_ofs << std::endl;
        return best_ofs;
    }

   private:
    double Cost(double offset, Eigen::Vector3d bias, int start_frame, int end_frame, double ou,
                double& der, Eigen::Vector3d& d_bias) {
        std::vector<int> frame_idxs;
        pair_storage_->GetFramesWith(frame_idxs, false, true, false, false, false);
        der = 0;
        d_bias = {0, 0, 0};
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
            double tmp;
            Eigen::Vector3d tmp2;
            cost += FrameCost(offset, bias, frame, ou, tmp, tmp2);
            der += tmp;
            d_bias += tmp2;
        }
        return cost;
    }

    double FrameCost(double offset, Eigen::Vector3d bias, int frame, double k, double& der,
                     Eigen::Vector3d& d_bias) {
        double rs_coeff = calibration_provider_->GetRsCoefficent();
        auto frame_height = calibration_provider_->GetCalibraiton().Height();
        PairDescription desc;
        pair_storage_->Get(frame, desc);

        int j = 0;
        for (int i = 0; i < desc.point_ids.size(); ++i) {
            if (!desc.mask_essential[i]) {
                continue;
            }
            ++j;
        }

        using ScalarT = Eigen::AutoDiffScalar<Eigen::Matrix<double, 4, 1>>;
        Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> problem(j, 3);
        j = 0;
        double interframe = desc.timestamp_b - desc.timestamp_a;
        for (int i = 0; i < desc.point_ids.size(); ++i) {
            double ts_a = rs_coeff * desc.points_a[i].y / frame_height,
                   ts_b = rs_coeff * desc.points_b[i].y / frame_height;

            if (!desc.mask_essential[i]) {
                continue;
            }

            auto gyro = integrator_.IntegrateGyro(
                (desc.timestamp_a + ts_a * interframe + offset) * sample_rate_,
                (desc.timestamp_b + ts_b * interframe + offset) * sample_rate_);
            auto gyro_lpf = integrator_lpf_.IntegrateGyro(
                (desc.timestamp_a + ts_a * interframe + offset) * sample_rate_,
                (desc.timestamp_b + ts_b * interframe + offset) * sample_rate_);
            auto gyrob = gyro.Bias(bias);
            auto gyrob_lpf = gyro_lpf.Bias(bias);

            Eigen::Vector3d point_a, point_b;
            point_a << desc.points_undistorted_a[i].x, desc.points_undistorted_a[i].y, 1;
            point_b << desc.points_undistorted_b[i].x, desc.points_undistorted_b[i].y, 1;

            auto drot_dt =
                (k * (gyrob.dt2 - gyrob.dt1) + (1 - k) * (gyrob_lpf.dt2 - gyrob_lpf.dt1)).eval();
            auto rot_wdt =
                (k * gyrob.rot.cast<ScalarT>() + (1 - k) * gyrob_lpf.rot.cast<ScalarT>()).eval();
            rot_wdt(0, 0).derivatives()(0, 0) = drot_dt(0, 0);
            rot_wdt(1, 0).derivatives()(0, 0) = drot_dt(1, 0);
            rot_wdt(2, 0).derivatives()(0, 0) = drot_dt(2, 0);
            rot_wdt(0, 0).derivatives().block<3, 1>(1, 0) =
                k * gyro.rot(0, 0).derivatives() + (1 - k) * gyro_lpf.rot(0, 0).derivatives();
            rot_wdt(1, 0).derivatives().block<3, 1>(1, 0) =
                k * gyro.rot(1, 0).derivatives() + (1 - k) * gyro_lpf.rot(1, 0).derivatives();
            rot_wdt(2, 0).derivatives().block<3, 1>(1, 0) =
                k * gyro.rot(2, 0).derivatives() + (1 - k) * gyro_lpf.rot(2, 0).derivatives();

            Eigen::Matrix<ScalarT, 3, 3> R = AngleAxisToRotationMatrix((-rot_wdt).eval());
            // Eigen::Matrix<ScalarT, 3, 3> R =
            // Eigen::AngleAxis<ScalarT>(-rot_wdt.norm(), rot_wdt.normalized())
            // .toRotationMatrix();
            Eigen::Matrix<ScalarT, 3, 1> point_br = R * point_b;
            point_br /= point_br(2, 0);
            point_br = (point_br - point_a) / (1 + ts_b - ts_a) + point_a;

            problem.row(j++) = point_br.normalized().cross(point_a.cast<ScalarT>().normalized());
        }

        auto svd = problem.jacobiSvd(Eigen::ComputeFullV);
        auto t = svd.matrixV().col(svd.matrixV().cols() - 1).normalized().eval();

        auto error = (problem * t).eval();

        auto cost = error.norm();
        der = cost.derivatives()(0, 0);
        d_bias = cost.derivatives().block<3, 1>(1, 0);
        return cost.value();
    }

    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
    std::shared_ptr<ICalibrationProvider> calibration_provider_;

    GyroIntegrator integrator_;
    GyroIntegrator integrator_lpf_;
    double sample_rate_;
};

void RegisterFineSync(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<FineSyncImpl>(ctx, name);
}

}  // namespace rssync