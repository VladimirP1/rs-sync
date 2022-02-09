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
        // LowpassGyro(gyro_data.data(), gyro_data.size(), sample_rate_ / 250.);

        integrator_ = {gyro_data.data(), static_cast<int>(gyro_data.size())};
    }

    double Run(double initial_offset, Eigen::Vector3d bias, double search_radius,
               double search_step, int start_frame, int end_frame) override {
        std::ofstream out{"fine_cost0.csv"};

        double ou = 1;
        double ofs = initial_offset, cost{};
        double m{}, v{}, g{};
        double b1{.9}, b2{.999}, eps{1e-8}, eta{.001};
        for (int i = 0; i < 80; ++i) {
            cost = Cost(ofs, bias, start_frame, end_frame, ou, g);
            m = b1 * m + (1 - b1) * g;
            v = b2 * v + (1 - b2) * (g * g);
            double m_ = m / (1 - b1);
            double v_ = v / (1 - b2);
            ofs = ofs - eta / (sqrt(v_) + eps) * m_;
            std::cout << "g=" << g << " ofs=" << ofs << std::endl; 

            if (i < 20) {
                ou = 1;
            } else if(i < 50) {
                double k = (i - 20.)/30;
                ou = k * .5 + (1 - k) * 1;
            } else {
                ou = .5;
            }
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
            double cost = Cost(ofs, bias, start_frame, end_frame, .9, der);
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
    double Cost(double offset, Eigen::Vector3d bias, int start_frame, int end_frame, double ou, double& der) {
        std::vector<int> frame_idxs;
        pair_storage_->GetFramesWith(frame_idxs, false, true, false, false, false);
        der = 0;
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
            cost += FrameCost(offset, bias, frame, ou, tmp);
            der += tmp;
        }
        return cost;
    }

    double FrameCost(double offset, Eigen::Vector3d bias, int frame, double ou, double& der) {
        double rs_coeff = calibration_provider_->GetRsCoefficent();
        auto frame_height = calibration_provider_->GetCalibraiton().Height();

        PairDescription desc;
        pair_storage_->Get(frame, desc);

        using ScalarT = Eigen::AutoDiffScalar<Eigen::Matrix<double, 1, 1>>;
        Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> problem(desc.point_ids.size(), 3);
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

            auto drot_dt = (gyrob.dt2 - gyrob.dt1).eval();
            auto rot_wdt = gyrob.rot.cast<ScalarT>().eval();
            rot_wdt(0, 0).derivatives()(0, 0) = drot_dt(0, 0);
            rot_wdt(1, 0).derivatives()(0, 0) = drot_dt(1, 0);
            rot_wdt(2, 0).derivatives()(0, 0) = drot_dt(2, 0);

            // TODO: handle this properly
            Eigen::Matrix<ScalarT, 3, 3> R =
                Eigen::AngleAxis<ScalarT>(-rot_wdt.norm(), rot_wdt.normalized()).toRotationMatrix();

            Eigen::Matrix<ScalarT, 3, 1> point_br = R * point_b;
            point_br /= point_br(2, 0);
            point_br = (point_br - point_a) / (1 + ts_b - ts_a) + point_a;

            problem.row(i) = point_br.normalized().cross(point_a.cast<ScalarT>().normalized());
        }

        auto svd = problem.jacobiSvd(Eigen::ComputeFullV);
        auto t = svd.matrixV().col(svd.matrixV().cols() - 1).normalized().eval();

        // Reweigh the rows based on error
        auto error = (problem * t).eval();

        std::vector<std::pair<ScalarT, int>> ps;
        for (int i = 0; i < error.rows(); ++i) {
            ps.push_back({error(i, 0), i});
        }
        std::sort(ps.begin(), ps.end());

        double k = ou;
        int points_left = static_cast<int>(desc.point_ids.size() * k);

        if (points_left < 3) {
            der = 0;
            return 0.;
        }

        Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> problem2(points_left, 3);
        for (int i = 0; i < problem2.rows(); ++i) {
            problem2.row(i) = problem.row(ps[i].second);
        }

        // Solve again
        svd = problem2.jacobiSvd(Eigen::ComputeFullV);
        t = svd.matrixV().col(svd.matrixV().cols() - 1).normalized();

        auto cost = (problem2 * t).norm();
        der = cost.derivatives()(0, 0);
        return cost.value();
    }

    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
    std::shared_ptr<ICalibrationProvider> calibration_provider_;

    GyroIntegrator integrator_;
    double sample_rate_;
};

void RegisterFineSync(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<FineSyncImpl>(ctx, name);
}

}  // namespace rssync