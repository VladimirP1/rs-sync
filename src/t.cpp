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

    ctx->ContextLoaded();

    ctx->GetComponent<ICorrelator>(kCorrelatorName)
        ->SetPatchSizes(cv::Size(40, 40), cv::Size(20, 20));
    // ctx->GetComponent<IGyroLoader>(kGyroLoaderName)
    //     ->SetOrientation(Quaternion<double>::FromRotationVector({-20.*M_PI/180.,0,0}));

    int pos = 120;
    for (int i = 30 * pos; i < 30 * pos + 30 * 5; ++i) {
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
        desc.enable_debug = true;
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
    }
    ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
        ->Run(.5, 1e-4, -100000, 100000);

    return 0;

    for (int i = 30 * pos; i < 30 * pos + 30 * 5; ++i) {
        ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(i);
        std::cout << i << std::endl;
        cv::Mat vis;
        if (ctx->GetComponent<IVisualizer>(kVisualizerName)->VisualizeCorrelations(vis, i)) {
            cv::imwrite("out" + std::to_string(i) + "a.jpg", vis);
        }
    }

    std::cout << "main done" << std::endl;

    return 0;
}
