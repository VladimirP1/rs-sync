#include "pose_estimator.hpp"

#include <numeric>
#include <opencv2/calib3d.hpp>

#include "calibration_provider.hpp"
#include "optical_flow.hpp"
#include "pair_storage.hpp"

#include <iostream>

namespace rssync {
class PoseEstimatorImpl : public IPoseEstimator {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        calibration_provider_ =
            ctx_.lock()->GetComponent<ICalibrationProvider>(kCalibrationProviderName);
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        optical_flow_ = ctx_.lock()->GetComponent<IOpticalFlow>(kOpticalFlowName);
    }

    bool EstimatePose(int frame_number) override {
        PairDescription desc;
        if (!pair_storage_->Get(frame_number, desc) || !desc.has_points ||
            desc.points_a.size() < 5) {
            optical_flow_->CalcOptflow(frame_number);
        }

        if (!pair_storage_->Get(frame_number, desc) || desc.points_a.size() < 5) {
            return false;
        }

        FisheyeCalibration calibration = calibration_provider_->GetCalibraiton();

        cv::fisheye::undistortPoints(desc.points_a, desc.points_undistorted_a,
                                     calibration.CameraMatrix(), calibration.DistortionCoeffs(),
                                     cv::Mat::eye(3, 3, CV_32F));
        cv::fisheye::undistortPoints(desc.points_b, desc.points_undistorted_b,
                                     calibration.CameraMatrix(), calibration.DistortionCoeffs(),
                                     cv::Mat::eye(3, 3, CV_32F));

        desc.has_undistorted = true;

        auto E =
            cv::findEssentialMat(desc.points_undistorted_a, desc.points_undistorted_b, 1.,
                                 cv::Point2d(0, 0), cv::RANSAC, .99, 5e-3, desc.mask_essential);

        if (E.rows != 3 || E.cols != 3) {
            pair_storage_->Update(frame_number, desc);
            return false;
        }

        desc.mask_4d = desc.mask_essential;

        cv::recoverPose(E, desc.points_undistorted_a, desc.points_undistorted_b,
                        cv::Mat::eye(3, 3, CV_64F), desc.R, desc.t, 1000, desc.mask_4d,
                        desc.points4d);

        // It is more convinient if all points have positive Z
        for (int i = 0; i < desc.points4d.cols; ++i) {
            if (desc.points4d(2, i) < 0) {
                desc.points4d.col(i) = -desc.points4d.col(i);
            }
        }

        desc.has_pose = true;
        desc.has_points4d = true;

        pair_storage_->Update(frame_number, desc);

        return true;
    }

   private:
    std::shared_ptr<ICalibrationProvider> calibration_provider_;
    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IOpticalFlow> optical_flow_;
};

void RegisterPoseEstimator(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<PoseEstimatorImpl>(ctx, name);
}
}  // namespace rssync