#include <iostream>
#include <stdexcept>
#include <thread>
#include <fstream>
#include <iomanip>

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
    RegisterCalibrationProvider(ctx, kCalibrationProviderName, "GoPro_Hero6_2160p_16by9_wide.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, kVisualizerName);
    RegisterNormalFitter(ctx, kNormalFitterName);
    RegisterCorrelator(ctx, kCorrelatorName);
    // RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA_fixed.CSV");
    RegisterGyroLoader(ctx, kGyroLoaderName, "GX019642.MP4.csv");
    // RegisterGyroLoader(ctx, kGyroLoaderName, "193653AA_FIXED.CSV");
    RegisterRoughGyroCorrelator(ctx, kRoughGyroCorrelatorName);
    RegisterFineGyroCorrelator(ctx, kFineGyroCorrelatorName);

    ctx->ContextLoaded();

    ctx->GetComponent<ICalibrationProvider>(kCalibrationProviderName)->SetRsCoefficent(.5);

    ctx->GetComponent<ICorrelator>(kCorrelatorName)
        ->SetPatchSizes(cv::Size(40, 40), cv::Size(20, 20));

    // Estimate OF and camera relative poses
    int i = 0;
    while (i < 17340) {
        std::cout << i << std::endl;

        cv::Mat frame;
        if (!ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i, frame)) {
            ++i;
            std::cout << "Cannot load frame" << std::endl;
            continue;
            // cv::imwrite("out.png", frame);
        } 

        ctx->GetComponent<IPoseEstimator>(kPoseEstimatorName)->EstimatePose(i);

        // if (i > 5248) {
        //     break;
        // }

        ctx->GetComponent<ICorrelator>(kCorrelatorName)->RefineOF(i);

        ++i;
    }

    std::ofstream out("sync.csv");
    out << std::setprecision(16) << std::fixed;
    for (int start = 0;; start += 60) {
        PairDescription desc;
        // if (!ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(start, desc)) {
        //     break;
        // }

        RoughCorrelationReport report;
        ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
            ->Run(0, .2, 1e-3, start, start + 120, &report);

        auto sync = ctx->GetComponent<IFineGyroCorrelator>(kFineGyroCorrelatorName)
                        ->Run(report.offset, report.bias_estimate, .03, 5e-4, start, start + 120);

        out << start << "," << sync << "," << report.offset << "," << report.bias_estimate(0,0) << "," << report.bias_estimate(0,1) << "," << report.bias_estimate(0,2) << std::endl;
        std::flush(out);
        std::cout << "Sync at " << start << " done" << std::endl;
    }

    std::cout << "main done" << std::endl;

    return 0;
}
