#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <bl/frame_loader.hpp>
#include <bl/utils.hpp>
#include <bl/optical_flow.hpp>
#include <bl/pair_storage.hpp>
#include <bl/calibration_provider.hpp>
#include <bl/pose_estimator.hpp>

using namespace rssync;

int main(int args, char** argv) {
    auto ctx = IContext::CreateContext();

    RegisterFrameLoader(ctx, kFrameLoaderName, "000458AA.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    RegisterCalibrationProvider(ctx, kCalibrationProviderName,
                                "hawkeye_firefly_x_lite_4k_43_v2.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);

    ctx->ContextLoaded();

    auto opt_flow = ctx->GetComponent<IOpticalFlow>(kOpticalFlowName);
    auto pair_storage = ctx->GetComponent<IPairStorage>(kPairStorageName);
    auto calibration_provider = ctx->GetComponent<ICalibrationProvider>(kCalibrationProviderName);
    auto pose_estimator = ctx->GetComponent<IPoseEstimator>(kPoseEstimatorName);

    auto calibration = calibration_provider->GetCalibraiton();

    std::ofstream out("dataset.txt");

    out << std::fixed << std::setprecision(16);

    int frame = 0;
    while (opt_flow->CalcOptflow(frame)) {
        pose_estimator->EstimatePose(frame);
        std::cout << "Calculated " << frame << std::endl;
        ++frame;
        // if (frame >= 10) break;
    }

    const double rs_coeff = .75;

    out << frame << "\n";
    frame = 0;
    for (PairDescription desc; pair_storage->Get(frame, desc); ++frame) {
        if (!desc.has_points || !desc.has_undistorted) {
            break;
        }

        out << desc.point_ids.size() << "\n";
        for (int i = 0; i < desc.point_ids.size(); ++i) {
            // clang-format off
            out
                << desc.points_undistorted_a[i].x << " " << desc.points_undistorted_a[i].y << " " 
                << desc.points_undistorted_b[i].x << " " << desc.points_undistorted_b[i].y << " "
                << rs_coeff * (desc.points_a[i].y / calibration.Height()) << " "
                << rs_coeff * (desc.points_b[i].y / calibration.Height()) << "\n";
            // clang-format on
        }
        out << "\n";

        std::cout << "Exported " << frame << std::endl;
    }

    return 0;
}
