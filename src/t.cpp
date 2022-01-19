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
#include <bl/correlator.hpp>

#include <ds/lru_cache.hpp>

#include <io/stopwatch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace rssync;

int main() {
    auto ctx = IContext::CreateContext();

    RegisterFrameLoader(ctx, kFrameLoaderName, "000458AA.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    RegisterCalibrationProvider(ctx, kCalibrationProviderName, "GoPro_Hero6_2160p_43.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, kVisualizerName);

    RegisterCorrelator(ctx, kCorrelatorName);

    ctx->ContextLoaded();

    ctx->GetComponent<ICorrelator>(kCorrelatorName)
        ->SetPatchSizes(cv::Size(20, 20), cv::Size(12, 12));

    for (int i = 30 * 38; i < 30 * 38 + 80; ++i) {
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

        ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(i);

        ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);
        for (int j = 0; j < desc.correlations.size(); ++j) {
            if (!desc.mask_correlation[j]) continue;

            //     cv::imwrite("out" + std::to_string(i) + "d" + std::to_string(j) + "a.jpg",
            //                 desc.debug_patches[j].first);
            //     cv::imwrite("out" + std::to_string(i) + "d" + std::to_string(j) + "b.jpg",
            //                 desc.debug_patches[j].second);
            auto ucorr = desc.correlations[j];
            // cv::extractChannel(ucorr, ucorr, 1);
            double min, max;
            cv::minMaxLoc(ucorr, &min, &max);
            ucorr -= min;
            ucorr /= (max - min);

            ucorr.convertTo(ucorr, CV_8UC1, 255);
            //     cv::cvtColor(ucorr, ucorr, cv::COLOR_GRAY2BGR);
            //     cv::resize(ucorr, ucorr, ucorr.size() * 6, 0, 0, cv::INTER_LINEAR);
            //     cv::applyColorMap(ucorr, ucorr, cv::COLORMAP_MAGMA);
            //     cv::circle(ucorr, {ucorr.cols / 2, ucorr.rows / 2}, 1, cv::Scalar(0, 255, 0), 1,
            //                cv::LINE_AA);
            cv::imwrite("out" + std::to_string(i) + "d" + std::to_string(j) + "c.jpg", ucorr);
        }

        cv::Mat vis;
        if (ctx->GetComponent<IVisualizer>(kVisualizerName)->VisualizeCorrelations(vis, i)) {
            cv::imwrite("out" + std::to_string(i) + "a.jpg", vis);
        }

        cv::Mat img;
        ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i + 1, img);
        img = img.clone();
        ctx->GetComponent<IVisualizer>(kVisualizerName)->DimImage(img, .4);
        ctx->GetComponent<IVisualizer>(kVisualizerName)->OverlayMatched(img, i, false);
        ctx->GetComponent<IVisualizer>(kVisualizerName)->OverlayMatchedTracks(img, i);
        cv::imwrite("out" + std::to_string(i) + ".jpg", img);
    }

    std::cout << "main done" << std::endl;

    return 0;
}
