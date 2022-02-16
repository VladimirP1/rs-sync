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
#include <bl/fine_sync.hpp>

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
#include <math/gyro_integrator.hpp>

using namespace rssync;

int main(int args, char** argv) {
    google::InitGoogleLogging(argv[0]);
    auto ctx = IContext::CreateContext();

    // RegisterFrameLoader(ctx, kFrameLoaderName, "DropMeFiles_tRyrZ/out-rs.mp4");
    // RegisterFrameLoader(ctx, kFrameLoaderName, "000458AA.MP4");
    RegisterFrameLoader(ctx, kFrameLoaderName, "GH011230.MP4");
    // RegisterFrameLoader(ctx, kFrameLoaderName, "GX019642.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    // RegisterCalibrationProvider(ctx, kCalibrationProviderName,
    // "hawkeye_firefly_x_lite_4k_43_v2.json");
    RegisterCalibrationProvider(ctx, kCalibrationProviderName, "GH011230.MP4.json");
    // RegisterCalibrationProvider(ctx, kCalibrationProviderName, "GoPro_Hero6_2160p_16by9_wide.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, kVisualizerName);
    // RegisterGyroLoader(ctx, kGyroLoaderName,
    // "DropMeFiles_tRyrZ/attic_without_fog_2_76_01_v2.csv");
    RegisterGyroLoader(ctx, kGyroLoaderName, "GH011230.MP4.csv");
    // RegisterGyroLoader(ctx, kGyroLoaderName, "GX019642.MP4.csv");
    // RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA_fixed.CSV");
    // RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA.bbl.csv");
    RegisterRoughGyroCorrelator(ctx, kRoughGyroCorrelatorName);
    RegisterFineSync(ctx, kFineSyncName);

    // ctx->GetComponent<IGyroLoader>(kGyroLoaderName)->SetOrientation({-5. * M_PI / 180., 0. *
    // M_PI / 180., 0});

    ctx->ContextLoaded();

    ctx->GetComponent<ICalibrationProvider>(kCalibrationProviderName)->SetRsCoefficent(0.34);
    // ctx->GetComponent<ICalibrationProvider>(kCalibrationProviderName)->SetRsCoefficent(0.416);

    auto gyro_loader = ctx->GetComponent<IGyroLoader>(kGyroLoaderName);

    std::vector<Eigen::Vector3d> gyro(gyro_loader->DataSize());
    gyro_loader->GetData(gyro.data(), gyro.size());

    auto sync_points = SuggestSyncPoints(gyro[0].data(), gyro.size(), 5, 10, 1000);

    std::ofstream out("sync.csv");

    for (auto sync_point : sync_points) {
        sync_point /= gyro_loader->SampleRate();
        int start = sync_point * 50 - 60;
        int end = sync_point * 50 + 60;
        std::cout << "Sync point: " << sync_point << std::endl;
        for (int frame = start; frame < end; ++frame) {
            std::cout << frame << std::endl;
            ctx->GetComponent<IPoseEstimator>(kPoseEstimatorName)->EstimatePose(frame);
        }

        // First rough sync
        RoughCorrelationReport rep0;
        ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
            ->Run(0, 1, 1e-2, start, end, &rep0);

        // Second rough sync
        RoughCorrelationReport rep1;
        ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
            ->Run(rep0.offset, .1, 1e-3, start, end, &rep1);

        // Fine sync
        auto sync = ctx->GetComponent<IFineSync>(kFineSyncName)
                        ->Run(rep1.offset, rep1.bias_estimate, start, end);

        out << sync_point << "," << sync << std::endl;
    }

    // RoughCorrelationReport rough_correlation_report, rep;
    // std::ofstream out("sync.csv");
    // ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
    //     ->Run(0, 1, 1e-2, -100000, 100000, &rough_correlation_report);
    // // int start = 30 * pos;
    // for (int start = 30 * pos; start < 30 * pos + 30 * 26; start += 30) {
    //     std::cout << start << std::endl;
    //     ctx->GetComponent<IRoughGyroCorrelator>(kRoughGyroCorrelatorName)
    //         ->Run(rough_correlation_report.offset, .1, 1e-3, start, start + 60, &rep);
    //     std::cout << rep.bias_estimate.transpose() << std::endl;
    //     {
    //         Stopwatch s("Sync");
    //         auto sync = ctx->GetComponent<IFineSync>(kFineSyncName)
    //                         ->Run(rep.offset, rep.bias_estimate, .03, 5e-4, start, start + 120);

    //         out << start << "," << sync << std::endl;
    //     }
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
