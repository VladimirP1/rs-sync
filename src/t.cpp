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

using namespace rssync;

class RsReprojector : public BaseComponent {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
    }

    struct MatchCtx{
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

    void Run(const RoughCorrelationReport& rough_report) {
        double kRsCooef = .75;
        double gyro_offset = rough_report.offset;
        PairDescription desc;
        pair_storage_->Get(38 * 30 + 5, desc);
        std::cout << desc.has_correlations << desc.has_points << desc.has_points4d << desc.has_pose
                  << std::endl;
        if (!desc.has_points || !desc.has_points4d || !desc.has_pose) {
            return;
        }
        double readout_duration = (desc.timestamp_b - desc.timestamp_a) * kRsCooef;
        double img_height_px = calibration_provider_->GetCalibraiton().Height();

        for (int i = 0; i < desc.point_ids.size(); ++i) {
            // We will need corr data and 4d pts
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
                            rough_report.bias_estimate);
            auto trans = desc.t * translation_scale;

            // cv::Mat_<double> tmp(3, 1, CV_64F);
            // cv::Mat P1 = cv::Mat::eye(4, 4, CV_64F);
            // cv::Mat P2 = cv::Mat::eye(4, 4, CV_64F);

            // tmp << rot.x(), rot.y(), rot.z();
            // cv::Rodrigues(tmp, P2(cv::Rect(0, 0, 3, 3)));
            // P2(cv::Rect(3, 0, 1, 3)) = trans;

            // cv::Mat P1inv, P2inv;
            // cv::invert(P1, P1inv);
            // cv::invert(P2, P2inv);

            // std::cout << P2inv << std::endl;

            // std::cout << (pt_b_timestamp - pt_a_timestamp) << " " << pt_a_timestamp << " "
            //           << pt_b_timestamp << std::endl;

            // std::cout << Bias(rot, rough_report.bias_estimate).ToRotationVector().norm() * 180 /
            // M_PI << std::endl;
        }
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

    int pos = 38;
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
        ->Run(rough_correlation_report.offset, .5, 1e-4, -100000, 100000, &rough_correlation_report);

    ctx->GetComponent<RsReprojector>("RsReprojector")->Run(rough_correlation_report);

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
