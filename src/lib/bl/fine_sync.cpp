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

    struct RegularizationResidual {
        const double k = 1e-2;
        template <class T>
        bool operator()(const T* const m, T* residual) const {
            residual[0] = m[0] * m[0] * k;
            residual[1] = m[1] * m[1] * k;
            residual[2] = m[2] * m[2] * k;
            return true;
        }
    };

    class FrameCostFunction2 : public ceres::DynamicCostFunction {
       public:
        FrameCostFunction2(FineSyncImpl* fs, int frame, double* ou)
            : fs_(fs), frame_(frame), ou_(ou) {
            AddParameterBlock(1);
            AddParameterBlock(3);
            AddParameterBlock(3);

            fs_->pair_storage_->Get(frame_, desc_);

            SetNumResiduals(1);
        }
        virtual ~FrameCostFunction2() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals,
                              double** jacobians) const override {
            double offset = parameters[0][0] / 1000.;
            Eigen::Vector3d bias, align;
            bias << parameters[1][0], parameters[1][1], parameters[1][2];
            align << parameters[2][0], parameters[2][1], parameters[2][2];

            bias.setZero();
            align.setZero();

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
                rot_wdt(0, 0).derivatives()(0, 0) = drot_dt(0, 0) * fs_->sample_rate_;
                rot_wdt(1, 0).derivatives()(0, 0) = drot_dt(1, 0) * fs_->sample_rate_;
                rot_wdt(2, 0).derivatives()(0, 0) = drot_dt(2, 0) * fs_->sample_rate_;
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
                problem.row(i).normalize();
            }

            // std::ofstream out("problem.xyz");
            // for (int i = 0; i < problem.rows(); ++i) {
            //     out << std::fixed << std::setprecision(16) << problem.row(i) << std::endl;
            // }
            // exit(0);

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
                    auto sm = der.block(0, 1, 1, 3).eval();
                    Eigen::Map<
                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        map(jacobians[1], sm.rows(), sm.cols());
                    map = sm / fs_->gyro_loader_->SampleRate();
                    map.setZero();
                }
                if (jacobians[2]) {
                    auto sm = der.block(0, 4, 1, 3).eval();
                    Eigen::Map<
                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        map(jacobians[2], sm.rows(), sm.cols());
                    map = sm;
                    map.setZero();
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

                Eigen::Vector3d best_sol;
                double least_med = std::numeric_limits<double>::infinity();
                srand(12);
                for (int i = 0; i < 200; ++i) {
                    int vs[3];
                    vs[0] = vs[1] = rand() % P.rows();
                    while (vs[1] == vs[0]) vs[2] = vs[1] = rand() % P.rows();
                    while (vs[2] == vs[1] || vs[2] == vs[0]) vs[2] = rand() % P.rows();

                    auto v = (Eigen::Vector3d{P.row(vs[0]) - P.row(vs[1])})
                                 .cross(Eigen::Vector3d{P.row(vs[0]) - P.row(vs[2])})
                                 .normalized()
                                 .eval();
                    auto residuals = (P * v).array().eval();
                    auto residuals2 = (residuals * residuals).eval();

                    std::sort(residuals2.data(), residuals2.data() + residuals2.rows());
                    // std::cout << residuals2.transpose() << std::endl;
                    auto med = residuals2(residuals2.rows() / 4, 0);

                    if (med < least_med) {
                        least_med = med;
                        best_sol = v;
                    }
                }
                // std::cout << best_sol.transpose() << std::endl;
                // -------------------------------
                auto residuals = (P * best_sol).array().eval();
                auto k = 1e1;
                weights = (1 / (1 + residuals * residuals * k * k)).sqrt().matrix();
                // weights = (1 / (1 + (residuals * k).cwiseAbs())).sqrt();


                // -------------------------------
                for (int i = 0; i < 10; ++i) {
                    auto svd = (P.array().colwise() * weights.array())
                                   .matrix()
                                   .jacobiSvd(Eigen::ComputeFullV);
                    auto V = svd.matrixV().eval();
                    residuals = (P * V.col(V.cols() - 1)).array().eval();

                    weights = (1 / (1 + residuals * residuals * k * k)).sqrt().matrix();
                    // weights = (1 / (1 + (residuals * k).cwiseAbs())).sqrt();
                }

                // weights *= residuals.matrix().norm();

                for (int i = 0; i < P_ad.rows(); ++i)
                    for (int j = 0; j < P_ad.cols(); ++j) P_ad(i, j) *= weights(i, 0);
            }
            // std::cout << weights.transpose() << std::endl;

            auto svd = P_ad.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
            auto U = svd.matrixU().eval();
            auto V = svd.matrixV().eval();
            auto S = svd.singularValues().eval();
            auto res = (U.col(U.cols() - 1) * S(S.rows() - 1, 0)).eval();

            residuals.resize(1, 1);
            for (int i = 0; i < 1; ++i) {
                residuals(i, 0) = S(S.rows() - 1, 0).value();  // res(i, 0).value();
            }

            const int n_der = P_ad(0, 0).derivatives().rows();
            derivatives.resize(1, n_der);
            for (int i = 0; i < 1; ++i) {
                for (int j = 0; j < n_der; ++j) {
                    derivatives(i, j) = S(S.rows() - 1, 0).derivatives()(j, 0);
                }
            }

            /* For explanation, see: https://j-towns.github.io/papers/svd-derivative.pdf */
            /*auto svd = P.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

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
            }*/
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
        std::vector<std::unique_ptr<FrameCostFunction2>> costs;
        double ou = 1;
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

            costs.emplace_back(std::make_unique<FrameCostFunction2>(this, frame, &ou));
        }
        Eigen::Vector3d align = {0, 0, 0};
        double* params[3];
        double* jac[3] = {0, 0, 0};
        for (double offset = initial_offset * 1000 - 50; offset < initial_offset * 1000 + 50;
             offset += .25) {
            params[0] = &offset;
            params[1] = bias.data();
            params[2] = align.data();
            double cst = 0, dcst = 0;
            for (auto& cf : costs) {
                Eigen::Matrix<double, -1, 1> res(cf->num_residuals(), 1);
                Eigen::Matrix<double, -1, 1> jac_t(cf->num_residuals(), 1);
                jac[0] = jac_t.data();
                cf->Evaluate(params, res.data(), jac);
                cst += res.norm();
                dcst += jac_t.sum();
            }
            out << offset << "," << cst << "," << dcst << std::endl;
        }
        // Eigen::Vector3d align = {0, 0, 0};
        // double* params[3];
        // double* jac[3] = {0, 0, 0};
        // int i = 0;
        // double offset = initial_offset;
        // // for (double offset = initial_offset * 1000 - 50; offset < initial_offset * 1000 + 50;
        //     //  offset += .25) {
        //     params[0] = &offset;
        //     params[1] = bias.data();
        //     params[2] = align.data();
        //     double cst = 0, dcst = 0;
        //     for (auto& cf : costs) {
        //         Eigen::Matrix<double, -1, 1> res(cf->num_residuals(), 1);
        //         Eigen::Matrix<double, -1, 1> jac_t(cf->num_residuals(), 1);
        //         jac[0] = jac_t.data();
        //         cf->Evaluate(params, res.data(), jac);
        //         out << i++ << "," << res.norm() << "," << jac_t.sum() << std::endl;
        //         cst += res.norm();
        //         dcst += jac_t.sum();
        //     }
        // // }
        return 0;
    }

    double Run(double initial_offset, Eigen::Vector3d bias, int start_frame,
               int end_frame) override {
        initial_offset *= 1000;

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
                                     nullptr, &initial_offset, bias.data(),
                                     align.data());
        }

        // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RegularizationResidual,3,3>(new
        // RegularizationResidual()), nullptr, bias.data());
        std::cout << "Before sync: " << initial_offset << "  " << bias.transpose() << std::endl;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 200;
        // options.parameter_tolerance = 1e-5;
        options.max_trust_region_radius = 1e-3;
        options.initial_trust_region_radius = 5e-4;
        options.use_inner_iterations = true;
        // options.use_nonmonotonic_steps = true;
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