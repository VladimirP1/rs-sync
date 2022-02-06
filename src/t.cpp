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
#include <bl/fine_gyro_correlator.hpp>

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

int main(int args, char** argv) {
    google::InitGoogleLogging(argv[0]);
    auto ctx = IContext::CreateContext();

    // RegisterFrameLoader(ctx, kFrameLoaderName, "000458AA.MP4");
    // RegisterFrameLoader(ctx, kFrameLoaderName, "193653AA.MP4");
    RegisterFrameLoader(ctx, kFrameLoaderName, "GX019642.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    // RegisterCalibrationProvider(ctx, kCalibrationProviderName,
                                // "hawkeye_firefly_x_lite_4k_43_v2.json");
    RegisterCalibrationProvider(ctx, kCalibrationProviderName,
    "GoPro_Hero6_2160p_16by9_wide.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, kVisualizerName);
    RegisterNormalFitter(ctx, kNormalFitterName);
    RegisterCorrelator(ctx, kCorrelatorName);
    // RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA_fixed.CSV");
    RegisterGyroLoader(ctx, kGyroLoaderName, "GX019642.MP4.csv");
    // RegisterGyroLoader(ctx, kGyroLoaderName, "193653AA_FIXED.CSV");
    RegisterRoughGyroCorrelator(ctx, kRoughGyroCorrelatorName);
    RegisterFineGyroCorrelator(ctx, kFineGyroCorrelatorName);
    // RegisterComponent<RsReprojector>(ctx, "RsReprojector");

    ctx->ContextLoaded();

    ctx->GetComponent<ICalibrationProvider>(kCalibrationProviderName)->SetRsCoefficent(.5);

    ctx->GetComponent<ICorrelator>(kCorrelatorName)
        ->SetPatchSizes(cv::Size(20, 20), cv::Size(10, 10));
    // ctx->GetComponent<IGyroLoader>(kGyroLoaderName)
    // ->SetOrientation(Quaternion<double>::FromRotationVector({-20. * M_PI / 180., 0, 0}));

    // int pos = 45;
    // int pos = 129;
    // int pos = 38;
    double pos = 85*2;
    // double pos = 6240./30;
    // double pos = 5555./30;
    // double pos = 5900./30;
    for (int i = 30 * pos; i < 30 * pos + 120; ++i) {
        std::cout << i << std::endl;

        ctx->GetComponent<IPoseEstimator>(kPoseEstimatorName)->EstimatePose(i);

        // PairDescription desc;
        // ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);
        // desc.enable_debug = true;
        // ctx->GetComponent<IPairStorage>(kPairStorageName)->Update(i, desc);

        ctx->GetComponent<ICorrelator>(kCorrelatorName)->RefineOF(i);

        // ctx->GetComponent<IPoseEstimator>(kPoseEstimatorName)->EstimatePose(i);

        // ctx->GetComponent<ICorrelator>(kCorrelatorName)->RefineOF(i);

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
    }
    // ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(38*30+5);
    // ctx->GetComponent<IVisualizer>(kVisualizerName)
    //     ->DumpDebugCorrelations(38 * 30 + 5, "corrs/out");

    RoughCorrelationReport rough_correlation_report, rep;
    std::ofstream out("sync.csv");
    ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
        ->Run(0, 1, 1e-1, -100000, 100000, &rough_correlation_report);
    int start = 30 * pos;
    // for (int start = 30 * pos; start < 30 * pos + 60*58; start += 60) {
    std::cout << start << std::endl;
    ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
        ->Run(rough_correlation_report.offset, .1, 1e-3, start, start + 120, &rep);

    auto sync = ctx->GetComponent<IFineGyroCorrelator>(kFineGyroCorrelatorName)
                    ->Run(rep.offset, rep.bias_estimate, .03, 5e-4, start, start + 120);

    out << start << "," << sync << std::endl;
    // }

    // std::cout << rough_correlation_report.bias_estimate.transpose() << std::endl;
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
