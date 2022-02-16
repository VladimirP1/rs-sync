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

#include <ceres/ceres.h>

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

    class FrameCostFunction2 : public ceres::DynamicCostFunction {
       public:
        FrameCostFunction2(FineSyncImpl* fs, int frame, double* ou)
            : fs_(fs), frame_(frame), ou_(ou) {
            AddParameterBlock(1);
            AddParameterBlock(3);
            AddParameterBlock(3);

            fs_->pair_storage_->Get(frame_, desc_);

            SetNumResiduals(desc_.point_ids.size());
        }
        virtual ~FrameCostFunction2() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals,
                              double** jacobians) const override {
            double offset = parameters[0][0] / 1000.;
            Eigen::Vector3d bias, align;
            bias << parameters[1][0], parameters[1][1], parameters[1][2];
            align << parameters[2][0], parameters[2][1], parameters[2][2];

            double rs_coeff = fs_->calibration_provider_->GetRsCoefficent();
            auto frame_height = fs_->calibration_provider_->GetCalibraiton().Height();

            using ScalarT = Eigen::AutoDiffScalar<Eigen::Matrix<double, 7, 1>>;
            Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> problem(desc_.point_ids.size(),
                                                                           3);
            double interframe = desc_.timestamp_b - desc_.timestamp_a;
            for (int i = 0; i < desc_.point_ids.size(); ++i) {
                double ts_a = rs_coeff * desc_.points_a[i].y / frame_height,
                       ts_b = rs_coeff * desc_.points_b[i].y / frame_height;

                auto gyro =
                    fs_->integrator_lpf_
                        .IntegrateGyro(
                            (desc_.timestamp_a + ts_a * interframe + offset) * fs_->sample_rate_,
                            (desc_.timestamp_b + ts_b * interframe + offset) * fs_->sample_rate_)
                        .Mix(fs_->integrator_.IntegrateGyro(
                                 (desc_.timestamp_a + ts_a * interframe + offset) *
                                     fs_->sample_rate_,
                                 (desc_.timestamp_b + ts_b * interframe + offset) *
                                     fs_->sample_rate_),
                             *ou_);

                auto gyrob = gyro.Bias(bias / fs_->gyro_loader_->SampleRate());

                Eigen::Vector3d point_a, point_b;
                point_a << desc_.points_undistorted_a[i].x, desc_.points_undistorted_a[i].y, 1;
                point_b << desc_.points_undistorted_b[i].x, desc_.points_undistorted_b[i].y, 1;

                auto drot_dt = (gyrob.dt2 - gyrob.dt1).eval();
                auto rot_wdt = gyrob.rot.cast<ScalarT>().eval();
                rot_wdt(0, 0).derivatives().setZero();
                rot_wdt(1, 0).derivatives().setZero();
                rot_wdt(2, 0).derivatives().setZero();
                rot_wdt(0, 0).derivatives()(0, 0) = drot_dt(0, 0);
                rot_wdt(1, 0).derivatives()(0, 0) = drot_dt(1, 0);
                rot_wdt(2, 0).derivatives()(0, 0) = drot_dt(2, 0);
                rot_wdt(0, 0).derivatives().block<3, 1>(1, 0) = gyro.rot(0, 0).derivatives();
                rot_wdt(1, 0).derivatives().block<3, 1>(1, 0) = gyro.rot(1, 0).derivatives();
                rot_wdt(2, 0).derivatives().block<3, 1>(1, 0) = gyro.rot(2, 0).derivatives();

                Eigen::Matrix<ScalarT, 3, 3> R_align =
                    AngleAxisToRotationMatrix(Eigen::Matrix<ScalarT, 3, 1>{
                        ScalarT{align(0, 0), 7, 4}, ScalarT{align(1, 0), 7, 5},
                        ScalarT{align(2, 0), 7, 6}});
                Eigen::Matrix<ScalarT, 3, 3> R = AngleAxisToRotationMatrix((-rot_wdt).eval());
                Eigen::Matrix<ScalarT, 3, 1> point_br =
                    (R_align * R * R_align.transpose()) * point_b;
                point_br /= point_br(2, 0);
                point_br = (point_br - point_a) / (1 + ts_b - ts_a) + point_a;

                problem.row(i) = point_br.normalized().cross(point_a.cast<ScalarT>().normalized());
            }

            double cost;
            Eigen::MatrixXd res, der;
            HlsSol(problem, res, der);

            std::copy(res.data(), res.data() + res.rows(), residuals);

            if (jacobians) {
                if (jacobians[0]) {
                    auto col = (der.col(0) / 1000.).eval();
                    std::copy(col.data(), col.data() + col.rows(), jacobians[0]);
                }
                if (jacobians[1]) {
                    auto sm = der.block(0, 1, der.rows(), 3).eval();
                    Eigen::Map<
                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        map(jacobians[1], sm.rows(), sm.cols());
                    map = sm / fs_->gyro_loader_->SampleRate();
                }
                if (jacobians[2]) {
                    auto sm = der.block(0, 4, der.rows(), 3).eval();
                    Eigen::Map<
                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        map(jacobians[2], sm.rows(), sm.cols());
                    map = sm;
                }
            }

            return true;
        }

       private:
        template <class ScalarT>
        void HlsSol(Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> P_ad,
                    Eigen::MatrixXd& residuals, Eigen::MatrixXd& derivatives) const {
            Eigen::MatrixXd P(P_ad.rows(), P_ad.cols());
            for (int j = 0; j < P.rows() * P.cols(); ++j) P.data()[j] = P_ad.data()[j].value();

            Eigen::Matrix<double, -1, 1> weights(P.rows(), 1);
            weights.setOnes();
            { /* https://hal.inria.fr/inria-00074015/document */
                // for (int i = 0; i < 10; ++i) {
                //     auto svd = (P.array().colwise() * weights.array())
                //                    .matrix()
                //                    .jacobiSvd(Eigen::ComputeFullV);
                //     auto V = svd.matrixV().eval();
                //     auto t = V.col(V.cols() - 1).eval();
                //     auto residuals = (P * t).array().eval();
                //     Eigen::Matrix<double, -1, 1> new_weights =
                //         1 / (1. + 100 * residuals.cwiseAbs());
                //     // if ((weights - new_weights).norm() < 1e-9) {
                //     //     weights = new_weights;
                //     //     break;
                //     // }
                //     weights = new_weights;
                // }
                // weights.normalize();
                // P = (P.array().colwise() * weights.array());
            }

            /* For explanation, see: https://j-towns.github.io/papers/svd-derivative.pdf */
            auto svd = P.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

            auto U = svd.matrixU().eval();
            auto S = svd.singularValues().eval();
            auto V = svd.matrixV().eval();
            auto S2 = S.cwiseProduct(S).eval();
            // clang-format off
            auto F =
                (1. / 
                    (
                        S2.transpose().replicate(S2.rows(), 1) - S2.replicate(1, S2.rows())
                    ).array()
                ).matrix().eval();
            F.diagonal().setZero();
            // clang-format on

            auto t = V.col(V.cols() - 1).eval();

            for (int j = 0; j < P.rows() * P.cols(); ++j) P.data()[j] = P_ad.data()[j].value();
            residuals = P * t;

            const int n_der = P_ad(0, 0).derivatives().rows();
            derivatives.resize(residuals.rows(), n_der);
            for (int i = 0; i < n_der; ++i) {
                Eigen::MatrixXd dP(P.rows(), P.cols());
                for (int j = 0; j < dP.rows() * dP.cols(); ++j)
                    dP.data()[j] = P_ad.data()[j].derivatives()(i, 0);
                dP = (dP.array().colwise() * weights.array());

                // clang-format off
                auto dV = 
                    V * (
                            (
                                F.array() * (
                                    S.asDiagonal() * U.transpose() * dP * V 
                                    + V.transpose() * dP.transpose() * U * S.asDiagonal()
                                ).array()
                            ).matrix() 
                            + (Eigen::MatrixXd::Identity(P.cols(), P.cols()) - V * V.transpose()) 
                                * dP.transpose() * U * S.asDiagonal().inverse()
                        );
                // clang-format on

                auto dt = dV.col(dV.cols() - 1).eval();
                derivatives.col(i) = dP * t + P * dt;
            }
        }

       private:
        FineSyncImpl* fs_;
        int frame_;
        double* ou_;

        PairDescription desc_;
    };

    double Run2(double initial_offset, Eigen::Vector3d bias, int start_frame,
                int end_frame) override {
        std::ofstream out("sync2.csv");
        double ou = 1;
        FrameCostFunction2 cost(this, start_frame, &ou);
        Eigen::Vector3d align = {0, 0, 0};
        double* params[3];
        for (double offset = initial_offset * 1000 - 50; offset < initial_offset * 1000 + 50;
             offset += .01) {
            params[0] = &offset;
            params[1] = bias.data();
            params[2] = align.data();
            Eigen::Matrix<double, -1, 1> res(cost.num_residuals(), 1);
            cost.Evaluate(params, res.data(), nullptr);
            out << offset << "," << res.norm() << std::endl;
        }
        return 0;
    }

    double Run(double initial_offset, Eigen::Vector3d bias, int start_frame,
               int end_frame) override {
        initial_offset *= 1000;
        bias *= gyro_loader_->SampleRate();

        ceres::Problem problem;
        double ou = 0;
        Eigen::Vector3d align;
        align.setZero();
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

            problem.AddResidualBlock(new FrameCostFunction2(this, frame, &ou),
                                     new ceres::CauchyLoss(.01), &initial_offset, bias.data(),
                                     align.data());
        }

        std::cout << "Before sync: " << initial_offset << "  " << bias.transpose() << std::endl;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 200;
        options.parameter_tolerance = 1e-5;
        options.max_trust_region_radius = 1e-3;
        options.initial_trust_region_radius = 5e-4;
        options.use_inner_iterations = true;
        options.use_nonmonotonic_steps = true;
        // options.check_gradients = true;
        // options.gradient_check_numeric_derivative_relative_step_size = 1e-8;

        options.num_threads = 8;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        std::cout << "Pre-Sync: " << initial_offset << "  " << bias.transpose() * 180 / M_PI * 200
                  << " | " << align.transpose() * 180 / M_PI << std::endl;

        ou = 1;

        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        std::cout << "Sync: " << initial_offset << "  " << bias.transpose() * 180 / M_PI * 200
                  << " | " << align.transpose() * 180 / M_PI << std::endl;

        return initial_offset;
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