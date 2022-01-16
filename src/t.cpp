#include <iostream>
#include <stdexcept>
#include <thread>

#include <bl/frame_loader.hpp>
#include <bl/utils.hpp>
#include <bl/optical_flow.hpp>
#include <ds/lru_cache.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <io/stopwatch.hpp>

namespace rssync {

}  // namespace rssync

int main() {
    auto ctx = rssync::IContext::CreateContext();

    rssync::RegisterFrameLoader(ctx, rssync::kFrameLoaderName, "141101AA.MP4");

    rssync::RegisterUuidGen(ctx, rssync::kUuidGenName);

    // rssync::RegisterKeypointDetector(ctx, rssync::kKeypointDetectorName);

    rssync::RegisterOpticalFlowLK(ctx, rssync::kOpticalFlowName);

    ctx->ContextLoaded();

    for (int i = 0; i < 800; ++i) {
        // cv::Mat out;
        // ctx->GetComponent<rssync::IFrameLoader>(rssync::kFrameLoaderName)->GetFrame(i, out);
        // std::cout << out.cols << std::endl;
        // rssync::OpticalFlowLK::KeypointInfo info;
        // ctx->GetComponent<rssync::OpticalFlowLK>("OpticalFlowLK")->GetKeypoints(i, info);
        ctx->GetComponent<rssync::IOpticalFlow>(rssync::kOpticalFlowName)->CalcOptflow(i);
        // std::cout << info.points.size() << std::endl;
    }

    std::cout << "main done" << std::endl;

    return 0;
}