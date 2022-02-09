#include "correlator.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <vision/camera_model.hpp>
#include <vision/utils.hpp>

#include <iostream>

#include "frame_loader.hpp"
#include "pair_storage.hpp"
#include "calibration_provider.hpp"
#include "normal_fitter.hpp"

namespace rssync {
class CorrelatorImpl : public ICorrelator {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        frame_loader_ = ctx_.lock()->GetComponent<IFrameLoader>(kFrameLoaderName);
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);
        normal_fitter_ = ctx_.lock()->GetComponent<INormalFitter>(kNormalFitterName);
    }

    void SetPatchSizes(cv::Size dst_a, cv::Size dst_b) override {
        dst_patch_size_a_ = dst_a;
        dst_patch_size_b_ = dst_b;
    }

    bool RefineOF(int frame_number) override {
        PairDescription desc;
        if (pair_storage_->Get(frame_number, desc); !desc.has_pose || !desc.has_points4d)
            return false;

        desc.mask_correlation = desc.mask_4d;
        // std::fill(desc.mask_correlation.begin(), desc.mask_correlation.end(), 1);

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

        // Resize destination arrays
        desc.patch_transforms.resize(desc.points4d.cols);
        if (desc.enable_debug) {
            desc.debug_correlations.resize(desc.points4d.cols);
            desc.debug_patches.resize(desc.points4d.cols);
        }

        for (int i = 0; i < desc.points4d.cols; ++i) {
            // Only process good points
            if (!desc.mask_correlation[i]) {
                continue;
            }

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
                                        P * Pa_inv * apx_undistort_a, dist_a, dst_patch_size_a_);
            success &=
                ExtractUndistortedPatch(patch_b, offset_map_b, frame_b,
                                        P * Pb_inv * apx_undistort_b, dist_b, dst_patch_size_b_);

            if (!success) {
                desc.mask_correlation[i] = false;
                continue;
            }

            // Compute correlation map between reprojected patches
            cv::Mat correlation_map;
            cv::matchTemplate(patch_b, patch_a, correlation_map, cv::TM_CCOEFF_NORMED);

            // Find best correlation
            double cx, cy;
            normal_fitter_->FindCenter(correlation_map, cx, cy);
            cx -= correlation_map.cols / 2. - .5;
            cy -= correlation_map.rows / 2. - .5;

            // Calculate new observed point position
            cv::Mat_<double> offset_map_b_inv, tmp_b{3, 1, CV_64F};
            cv::invert(offset_map_b, offset_map_b_inv);
            tmp_b << -cx, -cy, 1.;
            cv::Mat_<double> new_observed_b = offset_map_b_inv * tmp_b;

            new_observed_b.row(0) /= new_observed_b.row(2);
            new_observed_b.row(1) /= new_observed_b.row(2);

            // Maximum is at the edge
            if (std::isnan(new_observed_b(0, 0)) || std::isnan(new_observed_b(1, 0))) {
                desc.mask_correlation[i] = false;
                continue;
            }

            // Update distorted points
            desc.points_a[i] = dist_a;
            desc.points_b[i] = cv::Point2d{new_observed_b(0, 0), new_observed_b(1, 0)};

            // Update undistorted points
            FisheyeCalibration calibration = calibration_provider_->GetCalibraiton();

            cv::fisheye::undistortPoints(desc.points_a, desc.points_undistorted_a,
                                         calibration.CameraMatrix(), calibration.DistortionCoeffs(),
                                         cv::Mat::eye(3, 3, CV_32F));
            cv::fisheye::undistortPoints(desc.points_b, desc.points_undistorted_b,
                                         calibration.CameraMatrix(), calibration.DistortionCoeffs(),
                                         cv::Mat::eye(3, 3, CV_32F));

            desc.patch_transforms[i] = {offset_map_a, offset_map_b};
            if (desc.enable_debug) {
                desc.debug_correlations[i] = correlation_map;
                desc.debug_patches[i] = {patch_a, patch_b};
            }
        }

        pair_storage_->Update(frame_number, desc);

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
        cv::warpPerspective(frame(roi), patch, T_roi, dst_size, cv::INTER_LINEAR,
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
    std::shared_ptr<INormalFitter> normal_fitter_;

    cv::Size dst_patch_size_a_{20, 20}, dst_patch_size_b_{17, 17};
};

void RegisterCorrelator(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<CorrelatorImpl>(ctx, name);
}
}  // namespace rssync