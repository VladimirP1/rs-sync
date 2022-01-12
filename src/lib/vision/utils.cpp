#include "utils.hpp"

#include <opencv2/calib3d.hpp>

cv::Mat GetProjectionForUndistort(const FisheyeCalibration& calibration, bool projection) {
    cv::Mat P, ret{cv::Mat::eye(3, 4, CV_64F)};
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        calibration.CameraMatrix(), calibration.DistortionCoeffs(),
        cv::Size(calibration.Width(), calibration.Height()), cv::Mat::eye(3, 3, CV_64F), P);
    P.col(0).copyTo(ret.col(0));
    P.col(1).copyTo(ret.col(1));
    P.col(2).copyTo(ret.col(3));
    return projection ? ret : P;
}
