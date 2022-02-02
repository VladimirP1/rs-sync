#include "pose_estimator.hpp"

#include <numeric>
#include <opencv2/calib3d.hpp>

#include "calibration_provider.hpp"
#include "optical_flow.hpp"
#include "pair_storage.hpp"

#include <vision/ninepoint.hpp>

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

        auto points_undistorted_b_scaled = desc.points_undistorted_b;
#if 0
        constexpr double rs_cooef = .75;
        for (int i = 0; i < desc.point_ids.size(); ++i) {
            auto scale =
                (1 + rs_cooef * (desc.points_b[i].y - desc.points_a[i].y) / calibration.Height());
            points_undistorted_b_scaled[i] =
                (points_undistorted_b_scaled[i] - desc.points_undistorted_a[i]) / scale +
                desc.points_undistorted_a[i];
        }
#endif
#if 0
        auto E =
            cv::findEssentialMat(desc.points_undistorted_a, points_undistorted_b_scaled, 1.,
                                 cv::Point2d(0, 0), cv::RANSAC, .99, 5e-4, desc.mask_essential);
#endif
#if 1
        cv::Mat_<double> E(3, 3, CV_64F);
        {
            double k;
            std::vector<Eigen::Vector3d> points1, points2;
            for (int i = 0; i < desc.points_undistorted_a.size(); ++i) {
                Eigen::Vector3d p1, p2;
                p1 << desc.points_undistorted_a[i].x, desc.points_undistorted_a[i].y, (.75 * desc.points_a[i].y / calibration.Height());
                p2 << desc.points_undistorted_b[i].x, desc.points_undistorted_b[i].y, 1 + (.75 * desc.points_b[i].y / calibration.Height());
                points1.push_back(p1);
                points2.push_back(p2);
            }

            auto EE = FindEssentialMat(points1, points2, desc.mask_essential, 5e-7, 500, &k);

            std::cout << points1.size() << " / " << std::accumulate(desc.mask_essential.begin(), desc.mask_essential.end(), 0) << " -> " << k << std::endl;

            E << EE(0,0), EE(0,1), EE(0,2), EE(1,0), EE(1,1), EE(1,2), EE(2,0), EE(2,1), EE(2,2);
        }
#endif
        if (E.rows != 3 || E.cols != 3) {
            pair_storage_->Update(frame_number, desc);
            return false;
        }

        desc.mask_4d = desc.mask_essential;

        cv::recoverPose(E, desc.points_undistorted_a, points_undistorted_b_scaled,
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