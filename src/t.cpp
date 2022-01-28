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

class RsReprojector : public BaseComponent {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
    }

    struct MatchCtx {
        double rv[3];
        double t_scale;
        double xyz[3], w;
        double observed_a[2], observed_b[2];
    };

    struct PairCtx {
        double t[3];
        std::vector<MatchCtx> matches;
    };

    struct ProblemCtx {
        double focal[2], dist_params[6];
        std::vector<PairCtx> pairs;
    };

    struct CostFunctor {
        CostFunctor(ProblemCtx* ctx, PairCtx* pctx, MatchCtx* mctx)
            : ctx_{ctx}, pctx_{pctx}, mctx_{mctx} {}

        bool operator()(const double* point, const double* t, const double* lens_params,
                        double* residuals) const {
            Eigen::Matrix4d transformation;
            Eigen::Matrix4d transformation_j;
            transformation.setIdentity();

            // Rotation is constant for now...
            Eigen::Vector3d r{mctx_->rv[0], mctx_->rv[1], mctx_->rv[2]};
            transformation.block<3, 3>(0, 0) =
                Eigen::AngleAxis<double>(r.norm(), r.normalized()).toRotationMatrix();
            transformation.block<3, 1>(0, 3) = Eigen::Vector3d{t[0], t[1], t[2]};

            Eigen::Vector4d point4d = Eigen::Vector4d{point[0], point[1], point[2], mctx_->w};
            Eigen::Vector4d transformed = transformation * point4d;

            // std::cout << transformation << std::endl;
            // std::cout << "R" << r.norm() << transformed.transpose() << " " << point4d.transpose() << std::endl;
            // std::cout << "\n--------\n" << transformed.transpose() << "\n--------\n" << point4d.transpose() << std::endl;

            // Lens parameters
            double full_lens_params[8];
            full_lens_params[0] = ctx_->focal[0];
            full_lens_params[1] = ctx_->focal[1];
            std::copy_n(lens_params, 6, full_lens_params + 2);

            // Projection for view A
            double uv_a[2];
            double du_a[11], dv_a[11];
            ProjectPointJacobianExtended(point4d.data(), full_lens_params, uv_a, du_a, dv_a);

            // Projection for view B
            double uv_b[2];
            double du_b[11], dv_b[11];
            ProjectPointJacobianExtended(transformed.data(), full_lens_params, uv_b, du_b, dv_b);

            // Distance to observed
            double dist_x_a = mctx_->observed_a[0] - uv_a[0];
            double dist_y_a = mctx_->observed_a[1] - uv_a[1];

            double dist_x_b = mctx_->observed_b[0] - uv_b[0];
            double dist_y_b = mctx_->observed_b[1] - uv_b[1];

            // std::cout << "pp: " << full_lens_params[2] << " " << full_lens_params[3] << std::endl;
            // std::cout << "\nProjected:\n" << uv_a[0] << " " << uv_a[1] << "\n" << uv_b[0] << " " << uv_b[1] << std::endl;// << mctx_->observed_a[0] << " " << mctx_->observed_a[1] << " " << dist_x_a << " " << dist_y_a << std::endl;
            // std::cout << "\nObserved:\n" << mctx_->observed_a[0] << " " << mctx_->observed_a[1] << "\n" << mctx_->observed_b[0] << " " << mctx_->observed_b[1] << std::endl;// << mctx_->observed_a[0] << " " << mctx_->observed_a[1] << " " << dist_x_a << " " << dist_y_a << std::endl;

            residuals[0] = dist_x_a * dist_x_a + dist_y_a * dist_y_a;
            residuals[1] = dist_x_b * dist_x_b + dist_y_b * dist_y_b;

            // std::cout << residuals[0] << residuals[1] << std::endl;

            return true;
        }

       private:
        ProblemCtx* ctx_;
        PairCtx* pctx_;
        MatchCtx* mctx_;
    };

    void SolveProblem(ProblemCtx ctx) {
        ceres::Problem problem;
        for (auto& pctx : ctx.pairs) {
            for (auto& mctx : pctx.matches) {
                ceres::CostFunction* cost_function =
                    new ceres::NumericDiffCostFunction<CostFunctor, ceres::RIDDERS, 2, 3, 3, 6>(
                        new CostFunctor(&ctx, &pctx, &mctx));
                problem.AddResidualBlock(cost_function, nullptr, mctx.xyz, pctx.t, ctx.dist_params);
            }
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 250;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        for (int i = 0; i < 6; ++i) {
            std::cout << ctx.dist_params[i] << std::endl;
        }
        std::cout << std::endl;
        for (auto& f : ctx.pairs) {
            std::cout << Eigen::Vector3d{f.t[0], f.t[1], f.t[2]}.norm() << std::endl;
        }
    }

    ProblemCtx BuildProblem(const RoughCorrelationReport& rough_report) {
        ProblemCtx ctx;

        double kRsCooef = .75;
        double gyro_offset = rough_report.offset;

        auto calibration = calibration_provider_->GetCalibraiton();
        ctx.focal[0] = calibration.CameraMatrix()(0, 0);         // fx
        ctx.focal[1] = calibration.CameraMatrix()(1, 1);         // fy
        ctx.dist_params[0] = calibration.CameraMatrix()(0, 2);   // cx
        ctx.dist_params[1] = calibration.CameraMatrix()(1, 2);   // cy
        ctx.dist_params[2] = calibration.DistortionCoeffs()(0);  // k1
        ctx.dist_params[3] = calibration.DistortionCoeffs()(1);  // k2
        ctx.dist_params[4] = calibration.DistortionCoeffs()(2);  // k3
        ctx.dist_params[5] = calibration.DistortionCoeffs()(3);  // k4

        for (const auto& frame : rough_report.frames) {
            PairCtx pctx;

            PairDescription desc;
            pair_storage_->Get(frame, desc);

            pctx.t[0] = desc.t(0);
            pctx.t[1] = desc.t(1);
            pctx.t[2] = desc.t(2);

            double readout_duration = (desc.timestamp_b - desc.timestamp_a) * kRsCooef;
            double img_height_px = calibration_provider_->GetCalibraiton().Height();

            for (int i = 0; i < desc.point_ids.size(); ++i) {
                MatchCtx mctx;

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

                auto rot = Bias(gyro_loader_->GetRotation(pt_a_timestamp + gyro_offset,
                                                          pt_b_timestamp + gyro_offset),
                                rough_report.bias_estimate)
                               .ToRotationVector();

                mctx.rv[0] = rot[0];
                mctx.rv[1] = rot[1];
                mctx.rv[2] = rot[2];

                mctx.t_scale = translation_scale;

                mctx.xyz[0] = point4d[0];
                mctx.xyz[1] = point4d[1];
                mctx.xyz[2] = point4d[2];
                mctx.w = point4d[3];

                mctx.observed_a[0] = desc.points_a[i].x;
                mctx.observed_a[1] = desc.points_a[i].y;
                mctx.observed_b[0] = desc.points_b[i].x;
                mctx.observed_b[1] = desc.points_b[i].y;

                pctx.matches.push_back(mctx);

                // std::cout << (pt_b_timestamp - pt_a_timestamp) << " " << pt_a_timestamp << " "
                //           << pt_b_timestamp << std::endl;
            }

            ctx.pairs.push_back(pctx);
        }

        return ctx;
    }

   private:
    std::shared_ptr<ICalibrationProvider> calibration_provider_;
    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
};

int main() {
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
    RegisterRoughGyroCorrelator(ctx, kRoughGyroCorrelatorName);
    RegisterComponent<RsReprojector>(ctx, "RsReprojector");

    ctx->ContextLoaded();

    ctx->GetComponent<ICorrelator>(kCorrelatorName)
        ->SetPatchSizes(cv::Size(40, 40), cv::Size(20, 20));
    // ctx->GetComponent<IGyroLoader>(kGyroLoaderName)
    //     ->SetOrientation(Quaternion<double>::FromRotationVector({-20.*M_PI/180.,0,0}));

    int pos = 50;
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

    auto problem_ctx = ctx->GetComponent<RsReprojector>("RsReprojector")->BuildProblem(rough_correlation_report);
    ctx->GetComponent<RsReprojector>("RsReprojector")->SolveProblem(problem_ctx);

    std::cout << rough_correlation_report.frames.size() << std::endl;

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
