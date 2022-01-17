#include <iostream>
#include <stdexcept>
#include <thread>

#include <bl/frame_loader.hpp>
#include <bl/utils.hpp>
#include <bl/optical_flow.hpp>
#include <bl/pair_storage.hpp>
#include <bl/calibration_provider.hpp>
#include <bl/pose_estimator.hpp>
#include <bl/visualizer.hpp>


#include <ds/lru_cache.hpp>

#include <io/stopwatch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace rssync;


int main() {
    auto ctx = IContext::CreateContext();

    RegisterFrameLoader(ctx, kFrameLoaderName, "141101AA.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    RegisterCalibrationProvider(ctx, kCalibrationProviderName, "GoPro_Hero6_2160p_43.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, KVisualizerName);

    ctx->ContextLoaded();

    for (int i = 30*42; i < 30*42+800; ++i) {
        // cv::Mat out;
        // ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i, out);
        // std::cout << out.cols << std::endl;
        // OpticalFlowLK::KeypointInfo info;
        // ctx->GetComponent<OpticalFlowLK>("OpticalFlowLK")->GetKeypoints(i, info);
        // ctx->GetComponent<IOpticalFlow>(kOpticalFlowName)->CalcOptflow(i);
        ctx->GetComponent<IPoseEstimator>(kPoseEstimatorName)->EstimatePose(i);

        // PairDescription desc;
        // ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);
        // std::cout << desc.has_points << " " << desc.points_a.size() << " " << desc.t.at<double>(2) << std::endl;

        cv::Mat img;
        ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i + 1, img);
        img = img.clone();
        ctx->GetComponent<IVisualizer>(KVisualizerName)->DimImage(img, .4);
        ctx->GetComponent<IVisualizer>(KVisualizerName)->OverlayMatched(img, i, false);
        ctx->GetComponent<IVisualizer>(KVisualizerName)->OverlayMatchedTracks(img, i);
        cv::imwrite("out"+std::to_string(i)+".jpg",  img);
    }

    std::cout << "main done" << std::endl;

    return 0;
}

