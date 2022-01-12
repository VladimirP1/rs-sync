#include "utils.hpp"

#include <opencv2/calib3d.hpp>

cv::Mat GetProjectionForUndistort(const FisheyeCalibration& calibration) {
    cv::Mat P;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        calibration.CameraMatrix(), calibration.DistortionCoeffs(),
        cv::Size(calibration.Width(), calibration.Height()), cv::Mat::eye(3, 3, CV_64F), P);
    return P;
}

cv::Mat_<double> ProjectionTo2d(const cv::Mat_<double>& in, double z0, double w0, double z1) {
    cv::Mat_<double> out = cv::Mat::zeros(3, 3, CV_64F), tmp = cv::Mat::eye(3, 3, CV_64F);
    in.colRange(0, 2).rowRange(0, 3).copyTo(out.colRange(0, 2));
    out(0, 2) = (in(0, 2) * z0 + in(0, 3) * w0) / z1;
    out(1, 2) = (in(1, 2) * z0 + in(1, 3) * w0) / z1;
    out(2, 2) = (in(2, 2) * z0 + in(2, 3) * w0) / z1;
    out.colRange(0, 2) *= z0 / z1;
    return out;
}