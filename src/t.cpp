#include <iostream>
#include <stdexcept>
#include <thread>
#include <fstream>

#include <bl/frame_loader.hpp>
#include <bl/utils.hpp>
#include <bl/optical_flow.hpp>
#include <bl/pair_storage.hpp>
#include <bl/calibration_provider.hpp>
#include <bl/pose_estimator.hpp>
#include <bl/visualizer.hpp>
#include <bl/correlator.hpp>
#include <bl/normal_fitter.hpp>
#include <bl/gyro_loader.hpp>
#include <bl/rough_gyro_correlator.hpp>

#include <ds/lru_cache.hpp>

#include <io/stopwatch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <vision/camera_model.hpp>

using namespace rssync;

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;

Eigen::Vector3d GetGyroDerivative(IGyroLoader* loader, double time, double enlarge) {
    class CubicBcKernel {
       public:
        CubicBcKernel(double B = 0., double C = .5)
            : P0{(6. - 2. * B) / 6.},
              P1{0.},
              P2{(-18. + 12. * B + 6. * C) / 6.},
              P3{(12. - 9. * B - 6. * C) / 6.},
              Q0{(8. * B + 24. * C) / 6.},
              Q1{(-12. * B - 48. * C) / 6.},
              Q2{(6. * B + 30. * C) / 6.},
              Q3{(-1. * B - 6. * C) / 6.} {}

        double operator()(double x) const {
            if (x < 0) x = -x;
            if (x < 1.) return P0 + P1 * x + P2 * x * x + P3 * x * x * x;
            if (x < 2.) return Q0 + Q1 * x + Q2 * x * x + Q3 * x * x * x;
            return 0.;
        }

       private:
        double P0, P1, P2, P3, Q0, Q1, Q2, Q3;
    };
    static const CubicBcKernel krnl;

    double act_start, act_step;

    Eigen::Vector3d rvs[128];
    int d = std::ceil(5 * enlarge);
    if (d > 128) {
        abort();
    }
    loader->GetRawRvs(d / 2, time, act_start, act_step, rvs);

    Eigen::Vector3d rv{0, 0, 0};
    for (int i = 0; i < d; ++i) {
        double k = krnl((act_start + i * act_step - time) / act_step / enlarge);
        rv += rvs[i] * k;
    }
    // std::cout << (rv / act_step).norm() * 180 / M_PI << std::endl;
    return rv / act_step;
}

struct IntegrateGyroFunction : public ceres::SizedCostFunction<3, 2, 3> {
    IntegrateGyroFunction(IGyroLoader* loader) : gyro_loader_{loader} {}
    virtual bool Evaluate(double const* const* params, double* rotation, double** jacobians) const {
        double const* bounds = params[0];
        double const* bias = params[1];
        auto bias_vec = Eigen::Vector3d{bias};

        // Compute value
        auto rot_jet = gyro_loader_->GetRotation(bounds[0], bounds[1]);
        auto rot_val = Bias(rot_jet, bias_vec);
        Eigen::Map<Eigen::Vector3d>{rotation} = rot_val.ToRotationVector();

        // std::cout << Eigen::Vector3d{rotation} << "\npp\n" << rot_val << "\nqq\n" <<
        // rot_val.ToRotationVector() << "\n------------------------------------------\n";

        // Compute jacobians
        if (!jacobians) {
            return true;
        }

        // - for first parameter block, size 3x2
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> jac1{jacobians[0]};
            // jac1.block(0, 0, 1, 3) = GetGyroDerivative(gyro_loader_, bounds[0], expand_) +
            // bias_vec; jac1.block(1, 0, 1, 3) = GetGyroDerivative(gyro_loader_, bounds[1],
            // expand_) + bias_vec;
            jac1.fill(0);
        }

        // - for second parameter block, size 3x3
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac2{jacobians[1]};
            jac2 = BiasJacobian(rot_jet);
        }

        return true;
    }

   private:
    double expand_{1.};
    IGyroLoader* gyro_loader_;
};

struct ProjectPointFunction : public ceres::SizedCostFunction<2, 3, 8> {
    virtual bool Evaluate(double const* const* params, double* projected,
                          double** jacobians) const {
        double const* point = params[0];
        double const* lens_params = params[1];

        double uv[2], du[11], dv[11];
        ProjectPointJacobianExtended(point, lens_params, uv, du, dv);

        projected[0] = uv[0];
        projected[1] = uv[1];

        if (!jacobians) {
            return true;
        }

        // for first parameter block, 2x3
        if (jacobians[0]) {
            std::copy_n(du, 3, jacobians[0] + 0);
            std::copy_n(dv, 3, jacobians[0] + 3);
        }

        // for second parameter block, 2x8
        if (jacobians[1]) {
            std::copy_n(du + 3, 8, jacobians[1] + 0);
            std::copy_n(dv + 3, 8, jacobians[1] + 8);
        }

        return true;
    }
};

class RsReprojector : public BaseComponent {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
    }

    struct MatchCtx {
        double t_scale;
        Vector3d rv;
        Vector4d point;
        double point_z;
        Vector2d observed_a, observed_b;
        double ts_a, ts_b;
    };

    struct PairCtx {
        Vector3d tv;
        std::vector<MatchCtx> matches;
    };

    struct ProblemCtx {
        double gyro_delay;
        double rs_coeff;
        Vector3d gyro_bias;
        Matrix<double, 8, 1> lens_params;
        std::vector<PairCtx> pairs;
    };

    struct CostFunctor {
        CostFunctor(ceres::CostFunctionToFunctor<3, 2, 3>* integrate_gyro,
                    ceres::CostFunctionToFunctor<2, 3, 8>* project_point, ProblemCtx* ctx,
                    PairCtx* pctx, MatchCtx* mctx)
            : integrate_gyro_{integrate_gyro},
              project_point_{project_point},
              ctx_{ctx},
              pctx_{pctx},
              mctx_{mctx} {}

        template <typename T>
        bool operator()(const T* gyro_delay, const T* bias, const T* point_z, const T* translation,
                        T* residuals) const {
            // Point
            Matrix<T, 3, 1> zpoint;
            zpoint << T{mctx_->point[0]}, T{mctx_->point[1]}, T{mctx_->point[2]};
            zpoint = zpoint / mctx_->point[0] * point_z[0];

            // Integrate gyro
            Matrix<T, 2, 1> integration_segment;
            integration_segment << T{mctx_->ts_a} + gyro_delay[0], T{mctx_->ts_b} + gyro_delay[0];

            Matrix<T, 3, 1> rotation;
            (*integrate_gyro_)(integration_segment.data(), bias, rotation.data());

            // Rotate point
            Matrix<T, 3, 1> rotated_point;
            ceres::AngleAxisRotatePoint(rotation.data(), zpoint.data(), rotated_point.data());

            // Translate point
            rotated_point += Matrix<T, 3, 1>{translation} * mctx_->point[3] * mctx_->t_scale;

            // Cast lens_params to the appropriate type
            Matrix<T, 8, 1> lens_params0{ctx_->lens_params.cast<T>()};

            // Projection for view A
            // Matrix<T, 2, 1> uv_a;
            // (*project_point_)(zpoint.data(), lens_params0.data(), uv_a.data());

            // Projection for view B
            Matrix<T, 2, 1> uv_b;
            (*project_point_)(rotated_point.data(), lens_params0.data(), uv_b.data());

            // std::cout << uv_b << "\n-------------\n" << std::endl;
            // Distance to observed
            Matrix<T, 2, 1> observed_a{mctx_->observed_a.cast<T>()},
                observed_b{mctx_->observed_b.cast<T>()};

            // residuals[0] = (uv_a - observed_a).norm();
            residuals[0] = (uv_b - observed_b).norm();

            // std::cout << residuals[0]  << std::endl;

            return true;
        }

       private:
        ProblemCtx* ctx_;
        PairCtx* pctx_;
        MatchCtx* mctx_;
        ceres::CostFunctionToFunctor<3, 2, 3>* integrate_gyro_;
        ceres::CostFunctionToFunctor<2, 3, 8>* project_point_;
    };

    double SolveProblem(ProblemCtx& ctx) {
        ceres::Problem problem;
        ceres::CostFunctionToFunctor<3, 2, 3> integrate_gyro_functor{
            new IntegrateGyroFunction(gyro_loader_.get())};
        ceres::CostFunctionToFunctor<2, 3, 8> project_point_functor{new ProjectPointFunction()};
        int size = 0;
        for (auto& pctx : ctx.pairs) {
            for (auto& mctx : pctx.matches) {
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 3, 1, 3>(new CostFunctor(
                        &integrate_gyro_functor, &project_point_functor, &ctx, &pctx, &mctx));
                problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(.5), &ctx.gyro_delay,
                                         ctx.gyro_bias.data(), &mctx.point_z,
                                         pctx.tv.data());
                ++size;
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 1025;
        options.preconditioner_type = ceres::JACOBI;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.use_inner_iterations = true;
        options.use_nonmonotonic_steps = true;
        options.num_threads = 8;
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        std::cout << "Sync: " << ctx.gyro_delay << std::endl;
        std::cout << "Size: " << size << std::endl;

        // for (int i = 0; i < 6; ++i) {
        //     std::cout << ctx.dist_params[i] << std::endl;
        // }
        // std::cout << std::endl;
        // for (auto& f : ctx.pairs) {
        //     std::cout << Eigen::Vector3d{f.t[0], f.t[1], f.t[2]}.norm() << std::endl;
        // }
        return summary.final_cost;
    }

    ProblemCtx BuildProblem(const RoughCorrelationReport& rough_report) {
        ProblemCtx ctx;

        double gyro_offset = rough_report.offset;

        auto calibration = calibration_provider_->GetCalibraiton();
        ctx.lens_params << calibration.CameraMatrix()(0, 0), calibration.CameraMatrix()(1, 1),
            calibration.CameraMatrix()(0, 2), calibration.CameraMatrix()(1, 2),
            calibration.DistortionCoeffs()(0), calibration.DistortionCoeffs()(1),
            calibration.DistortionCoeffs()(2), calibration.DistortionCoeffs()(3);
        ctx.rs_coeff = .75;
        ctx.gyro_bias = rough_report.bias_estimate;
        ctx.gyro_delay = rough_report.offset;

        for (const auto& frame : rough_report.frames) {
            PairCtx pctx;

            PairDescription desc;
            pair_storage_->Get(frame, desc);

            pctx.tv << desc.t(0), desc.t(1), desc.t(2);

            double readout_duration = (desc.timestamp_b - desc.timestamp_a) * ctx.rs_coeff;
            double img_height_px = calibration_provider_->GetCalibraiton().Height();

            for (int i = 0; i < desc.point_ids.size(); ++i) {
                MatchCtx mctx;

                if (((size_t)random()) % 100 > 4) {
                    continue;
                }

                // We will need 4d point
                if (!desc.mask_4d[i]) {
                    continue;
                }

                double pt_a_timestamp =
                    desc.timestamp_a + readout_duration * (desc.points_a[i].y / img_height_px);
                double pt_b_timestamp =
                    desc.timestamp_b + readout_duration * (desc.points_b[i].y / img_height_px);
                double translation_scale =
                    (pt_b_timestamp - pt_a_timestamp) / (desc.timestamp_b - desc.timestamp_a);

                cv::Mat_<double> point4d_mat = desc.points4d.col(i);
                auto point4d = Matrix<double, 4, 1>(point4d_mat(0), point4d_mat(1), point4d_mat(2),
                                                    point4d_mat(3));

                if (point4d_mat(2) < .01) continue;

                mctx.rv = Bias(gyro_loader_->GetRotation(pt_a_timestamp + gyro_offset,
                                                         pt_b_timestamp + gyro_offset),
                               ctx.gyro_bias)
                              .ToRotationVector();
                mctx.t_scale = translation_scale;
                mctx.point = point4d;
                mctx.point_z = point4d[2];
                mctx.observed_a << desc.points_a[i].x, desc.points_a[i].y;
                mctx.observed_b << desc.points_b[i].x, desc.points_b[i].y;
                mctx.ts_a = pt_a_timestamp;
                mctx.ts_b = pt_b_timestamp;

                pctx.matches.push_back(mctx);

                // std::cout << (pt_b_timestamp - pt_a_timestamp) << " " << pt_a_timestamp << " "
                //           << pt_b_timestamp << std::endl;
            }

            ctx.pairs.push_back(pctx);
            // break;
        }

        return ctx;
    }

   private:
    std::shared_ptr<ICalibrationProvider> calibration_provider_;
    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
};

int main(int args, char** argv) {
    google::InitGoogleLogging(argv[0]);
    auto ctx = IContext::CreateContext();

    RegisterFrameLoader(ctx, kFrameLoaderName, "000458AA.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    RegisterCalibrationProvider(ctx, kCalibrationProviderName,
                                "hawkeye_firefly_x_lite_4k_43_v2.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, kVisualizerName);
    RegisterNormalFitter(ctx, kNormalFitterName);
    RegisterCorrelator(ctx, kCorrelatorName);
    RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA_fixed.CSV");
    // RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA.bbl.csv");
    RegisterRoughGyroCorrelator(ctx, kRoughGyroCorrelatorName);
    RegisterComponent<RsReprojector>(ctx, "RsReprojector");

    ctx->ContextLoaded();

    ctx->GetComponent<ICorrelator>(kCorrelatorName)
        ->SetPatchSizes(cv::Size(40, 40), cv::Size(20, 20));
    ctx->GetComponent<IGyroLoader>(kGyroLoaderName)
        ->SetOrientation(Quaternion<double>::FromRotationVector({-20. * M_PI / 180., 0, 0}));

    int pos = 42;
    for (int i = 30 * pos; i < 30 * pos + 30 * 2; ++i) {
        std::cout << i << std::endl;
        // cv::Mat out;
        // ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i, out);
        // std::cout << out.cols << std::endl;
        // OpticalFlowLK::KeypointInfo info;
        // ctx->GetComponent<OpticalFlowLK>("OpticalFlowLK")->GetKeypoints(i, info);
        // ctx->GetComponent<IOpticalFlow>(kOpticalFlowName)->CalcOptflow(i);
        ctx->GetComponent<IPoseEstimator>(kPoseEstimatorName)->EstimatePose(i);

        // PairDescription desc;
        // ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);
        // std::cout << desc.has_points << " " << desc.points_a.size() << " " <<
        // desc.t.at<double>(2) << std::endl;
        PairDescription desc;
        ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);
        desc.enable_debug = false;
        ctx->GetComponent<IPairStorage>(kPairStorageName)->Update(i, desc);

        ctx->GetComponent<ICorrelator>(kCorrelatorName)->RefineOF(i);

        // ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(i);

        // ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);

        // cv::Mat vis;
        // if (ctx->GetComponent<IVisualizer>(kVisualizerName)->VisualizeCorrelations(vis, i)) {
        //     cv::imwrite("out" + std::to_string(i) + "_.jpg", vis);
        // }

        // cv::Mat img;
        // ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i + 1, img);
        // img = img.clone();
        // ctx->GetComponent<IVisualizer>(kVisualizerName)->DimImage(img, .4);
        // ctx->GetComponent<IVisualizer>(kVisualizerName)->OverlayMatched(img, i, false);
        // ctx->GetComponent<IVisualizer>(kVisualizerName)->OverlayMatchedTracks(img, i);
        // cv::imwrite("out" + std::to_string(i) + ".jpg", img);

        // PairDescription desc;
        // ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);
        // double sum_corr = 0;
        // double count_corr = 0;
        // for (int i = 0; i < desc.correlation_models.size(); ++i) {
        //     if (desc.mask_correlation[i]) {
        //         sum_corr += desc.correlation_models[i].Evaluate(0, 0);
        //         count_corr += 1;
        //     }
        // }
        // std::cout << i << " " << sum_corr / count_corr << std::endl;
    }
    // ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(38*30+5);
    // ctx->GetComponent<IVisualizer>(kVisualizerName)
    //     ->DumpDebugCorrelations(38 * 30 + 5, "corrs/out");

    RoughCorrelationReport rough_correlation_report;

    ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
        ->Run(0, 40, 1e-2, -100000, 100000, &rough_correlation_report);

    ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
        ->Run(rough_correlation_report.offset, .5, 1e-4, -100000, 100000,
              &rough_correlation_report);
    std::cout << rough_correlation_report.frames.size() << std::endl;

    std::ofstream out("log.csv");
    double r_ofs = rough_correlation_report.offset;
    auto problem_ctx2 =
        ctx->GetComponent<RsReprojector>("RsReprojector")->BuildProblem(rough_correlation_report);
    ctx->GetComponent<RsReprojector>("RsReprojector")->SolveProblem(problem_ctx2);

    for (double ofs = r_ofs - .05; ofs < r_ofs + .03; ofs += .002) {
        // auto problem_ctx2 = problem_ctx;
        problem_ctx2.gyro_delay = ofs;
        double cost = ctx->GetComponent<RsReprojector>("RsReprojector")->SolveProblem(problem_ctx2);
        out << ofs << "," << cost << "," << problem_ctx2.gyro_delay << std::endl;
    }


    // for (int i = 30 * pos; i < 30 * pos + 30 * 5; ++i) {
    //     ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(i);
    //     std::cout << i << std::endl;
    //     cv::Mat vis;
    //     if (ctx->GetComponent<IVisualizer>(kVisualizerName)->VisualizeCorrelations(vis, i)) {
    //         cv::imwrite("out" + std::to_string(i) + "a.jpg", vis);
    //     }
    // }

    std::cout << "main done" << std::endl;

    return 0;
}
