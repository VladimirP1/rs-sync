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

        for (double ofs = initial_offset - 15e-3; ofs < initial_offset + 15e-3; ofs += 5e-5) {
            double cost = 0;
            for (int frame = start_frame; frame < end_frame; ++frame) {
                PairDescription desc;
                if (!pair_storage_->Get(frame, desc) || !desc.has_undistorted) continue;
                cost += std::fabs(FrameCost(frame, ofs, bias, {0, 0, 0}, 1.));
            }
            out << ofs * 1000 << "," << cost << std::endl;
        }

        return initial_offset;
    }

    double Run(double initial_offset, Eigen::Vector3d bias, int start_frame,
               int end_frame) override {
        double best_cost = std::numeric_limits<double>::infinity();
        double best_ofs_coarse = initial_offset;
        for (double ofs = initial_offset - 15e-3; ofs < initial_offset + 15e-3; ofs += 1e-3) {
            double cost = 0;
            for (int frame = start_frame; frame < end_frame; ++frame) {
                PairDescription desc;
                if (!pair_storage_->Get(frame, desc) || !desc.has_undistorted) continue;
                double residual = FrameCost(frame, ofs, bias, {0, 0, 0});
                // cost += log(1 + 10 * residual * residual);
                cost += residual * residual;
            }
            if (cost < best_cost) {
                best_cost = cost;
                best_ofs_coarse = ofs;
            }
        }
        double best_ofs_fine = best_ofs_coarse;
        for (double ofs = best_ofs_coarse - 2e-3; ofs < best_ofs_coarse + 2e-3; ofs += 5e-5) {
            double cost = 0;
            for (int frame = start_frame; frame < end_frame; ++frame) {
                PairDescription desc;
                if (!pair_storage_->Get(frame, desc) || !desc.has_undistorted) continue;
                double residual = FrameCost(frame, ofs, bias, {0, 0, 0});
                // cost += log(1 + 10 * residual * residual);
                cost += residual * residual;
            }
            if (cost < best_cost) {
                best_cost = cost;
                best_ofs_fine = ofs;
            }
        }

        std ::cout << "b-sync: " << best_ofs_fine << std::endl;
        return best_ofs_fine;
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

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> problem(desc_.point_ids.size(), 3),
            nproblem(desc_.point_ids.size(), 3);
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
            nproblem.row(i) = problem.row(i) / problem.row(i).norm();
            // double old_len = problem.row(i).norm();
            // problem.row(i).normalize();
            // nproblem.row(i) *= log(1 + old_len / 2e-2);
            // problem.row(i) *= atan(old_len / 2e-2);
            // problem.row(i) *= atan(old_len*M_PI/2)/M_PI*2;
            // std::cout << old_len << std::endl;
        }

        // std::ofstream out("problem.xyz");
        // for (int i = 0; i < problem.rows(); ++i) {
        //     out << std::fixed << std::setprecision(16) << problem.row(i) << std::endl;
        // }
        // exit(0);

        Eigen::Matrix<double, -1, 1> weights(nproblem.rows(), 1);
        weights.setOnes();
        { /* https://hal.inria.fr/inria-00074015/document */
            Eigen::Vector3d best_sol;
            double least_med = std::numeric_limits<double>::infinity();
            for (int i = 0; i < 200; ++i) {
                int vs[3];
                vs[0] = vs[1] = rand() % nproblem.rows();
                while (vs[1] == vs[0]) vs[2] = vs[1] = rand() % nproblem.rows();
                while (vs[2] == vs[1] || vs[2] == vs[0]) vs[2] = rand() % nproblem.rows();

                auto v = (Eigen::Vector3d{nproblem.row(vs[0]) - nproblem.row(vs[1])})
                             .cross(Eigen::Vector3d{nproblem.row(vs[0]) - nproblem.row(vs[2])})
                             .normalized()
                             .eval();

                // auto v = Eigen::Vector3d{normal(gen), normal(gen),
                // normal(gen)}.normalized().eval();
                auto residuals = (nproblem * v).array().eval();
                auto residuals2 = (residuals * residuals).eval();

                std::sort(residuals2.data(), residuals2.data() + residuals2.rows());
                auto med = residuals2(residuals2.rows() / 4, 0);
                if (med < least_med) {
                    least_med = med;
                    best_sol = v;
                }
            }
            auto residuals = (problem * best_sol).array().eval();
            auto k = 1e3;
            weights = (1 / (1 + residuals * residuals * k * k)).sqrt().matrix();
            // weights = (k / residuals.cwiseAbs()).cwiseMin(1).sqrt();
            // weights = (1 / (1 + (residuals * k).cwiseAbs())).sqrt();
            for (int i = 0; i < 20; ++i) {
                auto svd = (problem.array().colwise() * weights.array())
                               .matrix()
                               .jacobiSvd(Eigen::ComputeFullV);
                auto V = svd.matrixV().eval();
                residuals = (problem * V.col(V.cols() - 1)).array().eval();
                weights = (1 / (1 + residuals * residuals * k * k)).sqrt().matrix();
                // weights = (k / residuals.cwiseAbs()).cwiseMin(1).sqrt();
                // weights = (1 / (1 + (residuals * k).cwiseAbs())).sqrt();
            }

            problem = problem.array().colwise() * weights.array();
        }

        auto svd = problem.jacobiSvd();

        auto S = svd.singularValues().eval();

        // std::cout << S(S.rows() - 1, 0) << std::endl;
        // double tune_switch = 1e-1;
        // auto problem_weight = problem.rowwise().norm().sum() / tune_switch;
        // std::cout << problem_weight << std::endl;
        // double dir_weight = atan(problem_weight * M_PI / 2.) / M_PI * 2.;
        // return S(S.rows() - 1, 0) * dir_weight + problem_weight * (1 - dir_weight) * tune_switch;
        return S(S.rows() - 1, 0);
    }
};

void RegisterFineSync(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<FineSyncImpl>(ctx, name);
}

}  // namespace rssync