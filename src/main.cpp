#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <vision/calibration.hpp>

cv::Mat AffineApproximation(cv::Mat K, cv::Mat D, cv::Mat P, cv::Point2f center,
                            cv::Size patch_size) {
    static constexpr float winsize = 4;
    const auto patch_center =
        cv::Point2f{patch_size.width / 2.f, patch_size.height / 2.f};
    std::vector<cv::Point2f> to{},
        from{{center.x, center.y},
             {center.x - winsize, center.y - winsize},
             {center.x - winsize, center.y + winsize},
             {center.x + winsize, center.y + winsize},
             {center.x + winsize, center.y - winsize}};

    cv::fisheye::undistortPoints(from, to, K, D, cv::Mat::eye(3, 3, CV_64F), P);

    for (auto& p : to) {
        p -= to.front();
    }

    for (int i = 1; i < 3; ++i) to[i] = (to[i] - to[i + 2]) / 2.;

    for (int i = 0; i < 3; ++i) {
        from[i] = from[i] - center + patch_center;
        to[i] += patch_center;
    }

    return cv::getAffineTransform(from.data(), to.data());
}

cv::Mat GetP(const FisheyeCalibration& calibration) {
    cv::Mat P;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        calibration.CameraMatrix(), calibration.DistortionCoeffs(),
        cv::Size(calibration.Width(), calibration.Height()),
        cv::Mat::eye(3, 3, CV_64F), P);
    return P;
}

cv::Point2d UndistortPoint(cv::Point2d uv, cv::Mat camera_matrix,
                           cv::Mat distortion_cooficients) {
    static constexpr int kNumIterations = 9;
    auto& K = static_cast<cv::Mat_<double>&>(camera_matrix);
    auto& D = static_cast<cv::Mat_<double>&>(distortion_cooficients);

    double x_ = (uv.x - K(0, 2)) / K(0, 0);
    double y_ = (uv.y - K(1, 2)) / K(1, 1);

    double theta_ = std::sqrt(x_ * x_ + y_ * y_);

    double theta = M_PI / 4.;
    for (int i = 0; i < kNumIterations; ++i) {
        double theta2 = theta * theta, theta3 = theta2 * theta,
               theta4 = theta2 * theta2, theta5 = theta2 * theta3,
               theta6 = theta3 * theta3, theta7 = theta3 * theta4,
               theta8 = theta4 * theta4, theta9 = theta4 * theta5;
        double cur_theta_ = theta + D(0) * theta3 + D(1) * theta5 +
                            D(2) * theta7 + D(3) * theta9;
        double cur_dTheta_ = 1 + 3 * D(0) * theta2 + 5 * D(1) * theta4 +
                             7 * D(2) * theta6 + 8 * D(3) * theta8;
        double error = cur_theta_ - theta_;
        double new_theta = theta - error / cur_dTheta_;
        while (new_theta >= M_PI / 2. || new_theta <= 0.) {
            new_theta = (new_theta + theta) / 2.;
        }
        theta = new_theta;
    }

    double r = std::tan(theta);

    double s = (theta_ < 1e-9) ? 1. / std::cos(theta) : r / theta_;

    return cv::Point2d{x_ * s, y_ * s};
}

void UndistortPointG(cv::Point2d uv, cv::Point2d& xy, cv::Mat& J,
                     cv::Mat camera_matrix, cv::Mat distortion_cooficients) {
    static constexpr double eps = 1e-9;
    static constexpr int kNumIterations = 9;
    auto& K = static_cast<cv::Mat_<double>&>(camera_matrix);
    auto& D = static_cast<cv::Mat_<double>&>(distortion_cooficients);

    double x_ = (uv.x - K(0, 2)) / K(0, 0);
    double y_ = (uv.y - K(1, 2)) / K(1, 1);

    double dx_du = 1. / K(0, 0);
    double dy_dv = 1. / K(1, 1);

    double theta_ = std::sqrt(x_ * x_ + y_ * y_);
    double dtheta_dx_ = (theta_ < eps) ? 0 : 1. / theta_ * x_;
    double dtheta_dy_ = (theta_ < eps) ? 0 : 1. / theta_ * y_;
    double dtheta_dx_dx_ = (theta_ < eps) ? 0  // todo at 0
                                          : -1. / (theta_ * theta_ * x_ * x_) *
                                                (dtheta_dx_ * x_ + 1. * theta_);
    double dtheta_dx_dy_ = (theta_ < eps) ? 0  // todo at 0
                                          : -1. / (theta_ * theta_ * x_ * x_) *
                                                (dtheta_dy_ * x_ + 0. * theta_);
    double dtheta_dy_dx_ = (theta_ < eps) ? 0  // todo at 0
                                          : -1. / (theta_ * theta_ * y_ * y_) *
                                                (dtheta_dx_ * y_ + 0. * theta_);
    double dtheta_dy_dy_ = (theta_ < eps) ? 0  // todo at 0
                                          : -1. / (theta_ * theta_ * y_ * y_) *
                                                (dtheta_dx_ * y_ + 1. * theta_);

    double theta = M_PI / 4.;
    double dthetaDtheta_ = 0;
    double dthetaDtheta_Dtheta_ = 0;
    for (int i = 0; i < kNumIterations; ++i) {
        double theta2 = theta * theta, theta3 = theta2 * theta,
               theta4 = theta2 * theta2, theta5 = theta2 * theta3,
               theta6 = theta3 * theta3, theta7 = theta3 * theta4,
               theta8 = theta4 * theta4, theta9 = theta4 * theta5;
        double cur_theta_ = theta + D(0) * theta3 + D(1) * theta5 +
                            D(2) * theta7 + D(3) * theta9;
        double cur_dTheta_ = 1 + 3 * D(0) * theta2 + 5 * D(1) * theta4 +
                             7 * D(2) * theta6 + 8 * D(3) * theta8;
        double cur_Dtheta_Dtheta_ =
            2 * 3 * D(0) * theta + 4 * 5 * D(1) * theta3 +
            6 * 7 * D(2) * theta5 + 7 * 8 * D(3) * theta7;
        double error = cur_theta_ - theta_;
        dthetaDtheta_ = 1. / cur_dTheta_;
        dthetaDtheta_Dtheta_ = 1. / cur_Dtheta_Dtheta_;
        double new_theta = theta - error * dthetaDtheta_;
        while (new_theta >= M_PI / 2. || new_theta <= 0.) {
            new_theta = (new_theta + theta) / 2.;
        }
        theta = new_theta;
    }

    double r = std::tan(theta);
    double inv_cos_theta = 1. / std::cos(theta);
    double drDtheta = inv_cos_theta * inv_cos_theta;
    double drDthetaDtheta = 2 / (drDtheta * inv_cos_theta) * (std::sin(theta));

    double s = (theta_ < eps) ? inv_cos_theta : r / theta_;
    double drDtheta_ = drDtheta * dthetaDtheta_;
    double drDtheta_dtheta_ = drDthetaDtheta * dthetaDtheta_ * dthetaDtheta_ +
                              drDtheta * dthetaDtheta_Dtheta_;
    // clang-format off
    double dsDtheta_ =
        (theta_ < eps) ? 0. : (drDtheta_ * theta_ - r * 1) / theta_ / theta_;
    double dsDtheta_dtheta_ =
        (theta_ < eps) ? 0. : (drDtheta_dtheta_ * drDtheta_ - 1. * drDtheta_) / (theta_ * theta_) -
        (drDtheta_ * (theta_ * theta_) - 2. * 1. * r) / ((theta_ * theta_) * (theta_ * theta_));
    // clang-format on

    xy = {x_ * s, y_ * s};

    double dxdu = dx_du * s + x_ * dsDtheta_ * dtheta_dx_ * dx_du;
    double dydv = dy_dv * s + y_ * dsDtheta_ * dtheta_dy_ * dy_dv;
    double dxdv = x_ * dsDtheta_ * dtheta_dy_ * dy_dv;
    double dydu = y_ * dsDtheta_ * dtheta_dx_ * dx_du;

    // clang-format off
    double dxdudu = 
        2 * dx_du * dsDtheta_ * dtheta_dx_ * dx_du +
        x_ * (dsDtheta_dtheta_ * dtheta_dx_ * dx_du) * dtheta_dx_ * dx_du +
        x_ * dsDtheta_ * (dtheta_dx_dx_ * dx_du) * dx_du;
    double dydvdu = 
        dy_dv * dsDtheta_ * dtheta_dx_ * dx_du +
        y_ * (dsDtheta_dtheta_ * dtheta_dx_ * dx_du) * dtheta_dy_ * dy_dv +
        y_ * dsDtheta_ * (dtheta_dy_dx_ * dx_du) * dy_dv;
    double dxdvdu = 
        dx_du * dsDtheta_ * dtheta_dy_ * dy_dv +
        x_ * (dsDtheta_dtheta_ * dtheta_dx_ * dx_du) * dtheta_dy_ * dy_dv +
        x_ * dsDtheta_ * (dtheta_dy_dx_ * dx_du) * dy_dv;
    double dydudu = 
        y_ * (dsDtheta_dtheta_ * dtheta_dx_ * dx_du) * dtheta_dx_ * dx_du +
        y_ * dsDtheta_ * (dtheta_dx_dx_ * dx_du) * dx_du;
    
    // dx_du * dsDtheta_ * dtheta_dy_ * dy_dv
    double dxdudv = 
        dx_du * dsDtheta_ * dtheta_dy_ * dy_dv +
        x_ * (dsDtheta_dtheta_ * dtheta_dy_ * dy_dv) * dtheta_dx_ * dx_du +
        x_ * dsDtheta_ * (dtheta_dx_dy_ * dy_dv) * dx_du;
    double dydvdv = 
        2 * dy_dv * dsDtheta_ * dtheta_dy_ * dy_dv +
        y_ * (dsDtheta_dtheta_ * dtheta_dy_ * dy_dv) * dtheta_dy_ * dy_dv +
        y_ * dsDtheta_ * (dtheta_dy_dy_ * dy_dv) * dy_dv;
    double dxdvdv = 
        x_ * (dsDtheta_dtheta_ * dtheta_dy_ * dy_dv) * dtheta_dy_ * dy_dv +
        x_ * dsDtheta_ * (dtheta_dy_dy_ * dy_dv) * dy_dv;
    double dydudv = 
        dy_dv * dsDtheta_ * dtheta_dx_ * dx_du +
        y_ * (dsDtheta_dtheta_ * dtheta_dy_ * dy_dv) * dtheta_dx_ * dx_du +
        y_ * dsDtheta_ * (dtheta_dx_dy_ * dy_dv) * dx_du;
    // clang-format on

    std::cout << dxdudv << " " << dxdvdu << std::endl;

    cv::Mat_<double> ret(2, 2, CV_64F);
    ret << dxdu, dxdv, dydu, dydv;

    J = ret;
}

int main(int argc, char** argv) {
    std::ifstream fs("GoPro_Hero6_2160p_43.json");

    FisheyeCalibration c;
    fs >> c;
    auto P = GetP(c);

    cv::Mat J;
    cv::Point2d upoint;
    UndistortPointG(cv::Point2d{2005, 3000}, upoint, J, c.CameraMatrix(),
                    c.DistortionCoeffs());

    std::cout << upoint << std::endl << J * P.at<double>(0, 0) << std::endl;

    auto A = AffineApproximation(c.CameraMatrix(), c.DistortionCoeffs(), P,
                                 cv::Point2f{2005, 1508}, cv::Size(1, 1));

    std::cout << A << std::endl;

    // std::vector<cv::Point2d> upts, pts{cv::Point2d{2009, 1507}};
    // cv::fisheye::undistortPoints(pts, upts, c.CameraMatrix(),
    //                              c.DistortionCoeffs());
    // std::cout << upts[0] << std::endl;

    // cv::VideoCapture cap("141101AA.MP4");

    // cv::Mat frame;
    // cap.read(frame);

    // cv::Mat undistorted;
    // cv::fisheye::undistortImage(frame, undistorted, c.CameraMatrix(),
    // c.DistortionCoeffs(), P); cv::imwrite("outd.png", frame);
    // cv::imwrite("outu.png", undistorted);

    // cv::Mat result;
    // double patch_size = 100;
    // cv::Point2d center{3274, 445};
    // cv::Mat patch = frame(cv::Rect(center - cv::Point2d{patch_size,
    // patch_size},
    //                                cv::Size(patch_size * 2, patch_size *
    //                                2)));
    // auto A =
    //     AffineApproximation(c.CameraMatrix(), c.DistortionCoeffs(), P,
    //     center, patch.size());
    // cv::warpAffine(patch, result, A, cv::Size(patch_size * 2, patch_size *
    // 2));

    // cv::imwrite("outa.png", patch);
    // cv::imwrite("outb.png", result);
}