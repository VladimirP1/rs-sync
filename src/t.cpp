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

#include <vision/camera_model.hpp>
#include <vision/utils.hpp>

class Correlator : public BaseComponent {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        frame_loader_ = ctx_.lock()->GetComponent<IFrameLoader>(kFrameLoaderName);
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);
    }

    bool Calculate(int frame_number) {
        PairDescription desc;
        if (pair_storage_->Get(frame_number, desc); !desc.has_pose || !desc.has_points4d)
            return false;

        FisheyeCalibration calibration = calibration_provider_->GetCalibraiton();
        if (!calibration.IsLoaded()) return false;

        cv::Mat P = calibration_provider_->GetReasonableProjection();

        cv::Mat frame_a, frame_b;
        if (!frame_loader_->GetFrame(frame_number, frame_a)) return false;
        if (!frame_loader_->GetFrame(frame_number + 1, frame_b)) return false;

        // Construct projection matricies (inertial frame -> view A and B)
        cv::Mat_<double> Pa = cv::Mat::eye(3, 4, CV_64F);
        cv::Mat_<double> Pb(3, 4, CV_64F);
        Pb(cv::Rect(0, 0, 3, 3)) = desc.R * 1.;
        Pb.col(3) = desc.t * 1.;

        for (int i = 0; i < desc.points4d.cols; ++i) {
            // Only process good points
            if (!desc.mask_4d[i]) continue;

            // Apply projections to 4d points
            cv::Point3d point_a(cv::Mat(Pa * desc.points4d.col(i)));
            cv::Point3d point_b(cv::Mat(Pb * desc.points4d.col(i)));

            // Project the point through the camera model
            cv::Point2d dist_a, dist_b;
            cv::Mat_<double> apx_distort_a, apx_distort_b, apx_undistort_a, apx_undistort_b;
            ProjectPointJacobian(point_a, dist_a, apx_distort_a, calibration.CameraMatrix(),
                                 calibration.DistortionCoeffs());
            ProjectPointJacobian(point_b, dist_b, apx_distort_b, calibration.CameraMatrix(),
                                 calibration.DistortionCoeffs());

            // Calculate the linearized undistort transformation
            cv::invert(apx_distort_a, apx_undistort_a);
            cv::invert(apx_distort_b, apx_undistort_b);

            // Calculate local mapping (2d point in view A and B -> 3d point in inertial frame)
            cv::Mat Pa_inv, Pb_inv;
            cv::invert(To4x4(Pa), Pa_inv);
            cv::invert(To4x4(Pb), Pb_inv);

            // TODO: see if we can stop using point4d here, cause it is a bit wrong
            Pa_inv = ProjectionTo2d(Pa_inv, point_a.z, desc.points4d(3, i), 1.);
            Pb_inv = ProjectionTo2d(Pb_inv, point_b.z, desc.points4d(3, i), 1.);

            // Now we can map:
            //   distorted 2d points to undistorted 2d in camera frame [apx_undistort_*]
            //   undistorted 2d to 3d point in inertial frame          [P*_inv         ]
            //   3d point in inertial frame to virtual image plane     [P              ]

            cv::Mat patch_a, patch_b;
            cv::Mat offset_map_a, offset_map_b;
            bool success = true;
            success &=
                ExtractUndistortedPatch(patch_a, offset_map_a, frame_a,
                                        P * Pa_inv * apx_undistort_a, dist_a, cv::Size(20, 20));
            success &=
                ExtractUndistortedPatch(patch_b, offset_map_b, frame_b,
                                        P * Pb_inv * apx_undistort_b, dist_b, cv::Size(17, 17));

            if (!success) continue;

            // Compute correlation map between reprojected patches
            cv::Mat correlation_map;
            cv::matchTemplate(patch_a, patch_b, correlation_map, cv::TM_SQDIFF_NORMED);

            desc._debug_0_.push_back(correlation_map);
            desc._debug_1_.push_back(patch_b);
            pair_storage_->Update(frame_number, desc);
        }
        return true;
    }

   private:
    cv::Mat_<double> To4x4(const cv::Mat_<double>& in) {
        cv::Mat_<double> out = cv::Mat::eye(4, 4, CV_64F);
        in.copyTo(out(cv::Rect(cv::Point(0, 0), in.size())));
        return out;
    }

    bool ExtractUndistortedPatch(cv::Mat& patch, cv::Mat& offset_map, const cv::Mat& frame,
                                 cv::Mat transformation, cv::Point2d point_in_frame,
                                 cv::Size dst_size) {
        // Undistort patch center point
        cv::Mat_<double> point_m(3, 1, CV_64F);
        point_m << point_in_frame.x, point_in_frame.y, 1.;
        point_m = transformation * point_m;

        // Get the inverse transformation
        cv::Mat_<double> transformation_inv;
        cv::invert(transformation, transformation_inv);

        // See where the corners of the undistorted patch are in the distorted image
        cv::Mat_<double> points_d(3, 4, CV_64F);
        const double x_hs = dst_size.width / 2.;
        const double y_hs = dst_size.height / 2.;
        const double x = point_m(0), y = point_m(1), z = point_m(2);
        // clang-format off
        points_d << 
            x - x_hs * z, x + x_hs * z, x + x_hs * z, x - x_hs * z,
            y - y_hs * z, y - y_hs * z, y + y_hs * z, y + y_hs * z,
            z           , z           , z           , z           ;
        // clang-format on
        points_d = transformation_inv * points_d;

        // The distorted points will not necceserly lie in Z=1
        // Return them into Z=1
        points_d.row(0) /= points_d.row(2);
        points_d.row(1) /= points_d.row(2);

        // Find their bounding box
        double min_x, min_y, max_x, max_y;
        min_x = min_y = std::numeric_limits<double>::max();
        max_x = max_y = std::numeric_limits<double>::min();
        for (int i = 0; i < 4; ++i) {
            double x = points_d(0, i), y = points_d(1, i);
            min_x = std::min(x, min_x);
            min_y = std::min(y, min_y);
            max_x = std::max(x, max_x);
            max_y = std::max(y, max_y);
        }

        // Round the bounding box
        min_x = std::floor(min_x);
        min_y = std::floor(min_y);
        max_x = std::ceil(max_x);
        max_y = std::ceil(max_y);

        // Build ROI for source image
        auto roi = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);

        // Check if ROI is inside source image
        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width >= frame.cols ||
            roi.y + roi.height >= frame.rows) {
            return false;
        }

        // Retarget transformation from source frame to ROI
        cv::Mat_<double> T_roi = transformation.clone();
        cv::Mat_<double> adjustment = cv::Mat::eye(3, 3, CV_64F);
        adjustment.col(2) << roi.x, roi.y, 1.;
        T_roi = T_roi * adjustment;

        // Retarget from destination undistorted frame to dest ROI
        adjustment.col(2) << x_hs - x / z, y_hs - y / z, 1;
        T_roi = adjustment * T_roi;

        // Remap
        cv::warpPerspective(frame(roi), patch, T_roi, dst_size, cv::INTER_CUBIC,
                            cv::BORDER_CONSTANT, cv::Scalar(0, 255, 0));

        // Remember mapping for future correlation lookups
        adjustment.col(2) << -x / z, -y / z, 1;
        offset_map = adjustment * transformation;

        return true;
    }

   private:
    std::shared_ptr<IFrameLoader> frame_loader_;
    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<ICalibrationProvider> calibration_provider_;
};

int main() {
    auto ctx = IContext::CreateContext();

    RegisterFrameLoader(ctx, kFrameLoaderName, "000458AA.MP4");
    RegisterUuidGen(ctx, kUuidGenName);
    RegisterPairStorage(ctx, kPairStorageName);
    RegisterOpticalFlowLK(ctx, kOpticalFlowName);
    RegisterCalibrationProvider(ctx, kCalibrationProviderName, "GoPro_Hero6_2160p_43.json");
    RegisterPoseEstimator(ctx, kPoseEstimatorName);
    RegisterVisualizer(ctx, kVisualizerName);

    RegisterComponent<Correlator>(ctx, "debug0");

    ctx->ContextLoaded();

    for (int i = 30 * 31; i < 30 * 31 + 80; ++i) {
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

        ctx->GetComponent<Correlator>("debug0")->Calculate(i);

        PairDescription desc;
        ctx->GetComponent<IPairStorage>(kPairStorageName)->Get(i, desc);
        for (int j = 0; j < desc._debug_0_.size(); ++j) {
            cv::imwrite("out" + std::to_string(i) + "d" + std::to_string(j) + "a.jpg",
                        desc._debug_1_[j]);
            auto ucorr = desc._debug_0_[j];
            double min, max;
            cv::minMaxLoc(ucorr, &min, &max);
            ucorr -= min;
            ucorr /= (max - min);

            ucorr.convertTo(ucorr, CV_8UC1, 255);
            cv::cvtColor(ucorr, ucorr, cv::COLOR_GRAY2BGR);
            cv::resize(ucorr, ucorr, ucorr.size() * 6, 0, 0, cv::INTER_LINEAR);
            cv::applyColorMap(ucorr, ucorr, cv::COLORMAP_MAGMA);
            cv::circle(ucorr, {ucorr.cols / 2, ucorr.rows / 2}, 1, cv::Scalar(0, 255, 0), 1,
                       cv::LINE_AA);
            cv::imwrite("out" + std::to_string(i) + "d" + std::to_string(j) + "b.jpg",
                        ucorr);
        }

        cv::Mat img;
        ctx->GetComponent<IFrameLoader>(kFrameLoaderName)->GetFrame(i + 1, img);
        img = img.clone();
        // ctx->GetComponent<IVisualizer>(KVisualizerName)->DimImage(img, .4);
        // ctx->GetComponent<IVisualizer>(KVisualizerName)->OverlayMatched(img, i, false);
        // ctx->GetComponent<IVisualizer>(KVisualizerName)->OverlayMatchedTracks(img, i);
        cv::imwrite("out" + std::to_string(i) + ".jpg", img);
    }

    std::cout << "main done" << std::endl;

    return 0;
}
