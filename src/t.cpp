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

#include <ds/lru_cache.hpp>

#include <io/stopwatch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

using namespace rssync;

class GyroRoughCorrelator : public BaseComponent {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
    }

    void Run() {
        std::ofstream out("data.csv");
        std::vector<int> good_frames;
        pair_storage_->GetFramesWith(good_frames, false, false, true, false, false);
        std::sort(good_frames.begin(), good_frames.end());

        QuaternionGroup grp;
        cv::Mat_<double> rv;
        PairDescription desc;
        std::vector<std::tuple<double, double, Quaternion, double>> data;
        for (auto frame : good_frames) {
            pair_storage_->Get(frame, desc);
            cv::Rodrigues(desc.R, rv);
            data.emplace_back(desc.timestamp_a, desc.timestamp_b, Quaternion{rv(0), rv(1), rv(2)},
                              desc.points_a.size());
            std::cout << desc.timestamp_a << desc.timestamp_b << std::endl;
        }

        double min_cost = std::numeric_limits<double>::max();
        double best_shift = 0.;
        for (double shift = -.5; shift < .5; shift += 1e-4) {
            double cost = 0;
            for (auto frame_info : data) {
                double x, y, z;
                auto of_rot = std::get<2>(frame_info);
                auto gyro_rot = gyro_loader_->GetRotation(std::get<0>(frame_info) + shift,
                                                          std::get<1>(frame_info) + shift);

                // of_rot.ToRotVec(x, y, z);
                // out << x << "," << y << "," << z << ",";
                // gyro_rot.ToRotVec(x, y, z);
                // out << x << "," << y << ","<< z << ",";

                grp.add(grp.inv(gyro_rot), of_rot).ToRotVec(x, y, z);
                double residual = x * x + y * y + z * z;
                cost += log(1. + residual * 100.);
                // out << std::get<3>(frame_info) << "\n";
            }
            out << shift << "," << cost/data.size() << "\n" << std::endl;
            // break;
            // std::cout << cost/data.size() << " " << shift << std::endl;
            if (cost / data.size() < min_cost) {
                min_cost = cost / data.size();
                best_shift = shift;
            }
        }
        std::cout << "Sync: " << best_shift << std::endl;

        for (auto frame : good_frames) {
            pair_storage_->Get(frame, desc);
            auto gyro_rot = gyro_loader_->GetRotation(desc.timestamp_a + best_shift,
                                                      desc.timestamp_b + best_shift);
            double x, y, z;
            cv::Mat_<double> rv(3, 1, CV_64F);
            gyro_rot.ToRotVec(x, y, z);
            rv << x, y, z;
            cv::Rodrigues(rv, desc.R);
            // desc.t << 0,0,0;
            pair_storage_->Update(frame, desc);
        }

        std::cout << "Rotations updated" << std::endl;
    }

   private:
    double GetRotMagnitude(const Quaternion& q) {
        double x, y, z;
        q.ToRotVec(x, y, z);
        double norm = sqrt(x * x + y * y + z * z);
        return norm;
    }

   private:
    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
};

int main() {
    auto ctx = IContext::CreateContext();

    RegisterFrameLoader(ctx, kFrameLoaderName, "000458AA.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    RegisterCalibrationProvider(ctx, kCalibrationProviderName, "hawkeye_firefly_x_lite_4k_43_v2.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, kVisualizerName);
    RegisterNormalFitter(ctx, kNormalFitterName);
    RegisterCorrelator(ctx, kCorrelatorName);
    RegisterGyroLoader(ctx, kGyroLoaderName, "000458AA_fixed_nodc.CSV");
    RegisterComponent<GyroRoughCorrelator>(ctx, "GyroRoughCorrelator");

    ctx->ContextLoaded();

    ctx->GetComponent<ICorrelator>(kCorrelatorName)
        ->SetPatchSizes(cv::Size(40, 40), cv::Size(20, 20));
    int pos = 38;
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
        desc.enable_debug = false;
        ctx->GetComponent<IPairStorage>(kPairStorageName)->Update(i, desc);

        ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(i);

        // ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);

        cv::Mat vis;
        if (ctx->GetComponent<IVisualizer>(kVisualizerName)->VisualizeCorrelations(vis, i)) {
            cv::imwrite("out" + std::to_string(i) + "_.jpg", vis);
        }

        cv::Mat img;
        ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i + 1, img);
        img = img.clone();
        ctx->GetComponent<IVisualizer>(kVisualizerName)->DimImage(img, .4);
        ctx->GetComponent<IVisualizer>(kVisualizerName)->OverlayMatched(img, i, false);
        ctx->GetComponent<IVisualizer>(kVisualizerName)->OverlayMatchedTracks(img, i);
        cv::imwrite("out" + std::to_string(i) + ".jpg", img);
    }
    ctx->GetComponent<GyroRoughCorrelator>("GyroRoughCorrelator")->Run();

    for (int i = 30 * pos; i < 30 * pos + 30 * 5; ++i) {
        ctx->GetComponent<ICorrelator>(kCorrelatorName)->Calculate(i);

        cv::Mat vis;
        if (ctx->GetComponent<IVisualizer>(kVisualizerName)->VisualizeCorrelations(vis, i)) {
            cv::imwrite("out" + std::to_string(i) + "a.jpg", vis);
        }
    }

    std::cout << "main done" << std::endl;

    return 0;
}
