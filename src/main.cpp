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

    double f_x = K(0, 0), f_y = K(1, 1), c_x = K(0, 2), c_y = K(1, 2);
    double u = uv.x, v = uv.y;

    // clang-format off
    double x0 = pow(f_x, -2);
    double x1 = c_x - u;
    double x2 = x0*pow(x1, 2);
    double x3 = pow(f_y, -2);
    double x4 = c_y - v;
    double x5 = x3*pow(x4, 2);
    double x6 = x2 + x5;
    double x7 = sqrt(x6);
    
    double theta_ = x7;
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

    double x8 = tan(theta);
    double x9 = x8/x7;
    double x10 = 1.0/f_x;
    double x11 = x1*x10;
    double x12 = 1.0/f_y;
    double x13 = x12*x4;
    double x14 = pow(x6, -3.0/2.0);
    double x15 = x14*x8;
    double x16 = x15*x2;
    double x17 = pow(x8, 2) + 1;
    double x18 = theta;
    double x19 = dthetaDtheta_;
    double x20 = x17*x19;
    double x21 = x20/x6;
    double x22 = -x15 + x21;
    double x23 = x3*x4;
    double x24 = x0*x13;
    double x25 = x15*x5;
    double x26 = 3*x15 - 3*x21;
    double x27 = 3*x2;
    double x28 = x8/pow(x6, 5.0/2.0);
    double x29 = x14*x17*dthetaDtheta_Dtheta_;
    double x30 = x20/pow(x6, 2);
    double x31 = 2*x17*pow(x19, 2);
    double x32 = -x16*x31 - x2*x29 - x27*x28 + x27*x30;
    double x33 = x15 - x21;
    double x34 = x32 + x33;
    double x35 = x10*x23*x34;
    double x36 = 3*x5;
    double x37 = -x25*x31 - x28*x36 - x29*x5 + x30*x36;
    double x38 = x33 + x37;
    double x39 = x0*x1*x12*x38;

    double x = -x11*x9;
    double y = -x13*x9;
    double dxdu = x10*(-x16 + x2*x21 + x9);
    double dxdv = x11*x22*x23;
    double dydu = x1*x22*x24;
    double dydv = x12*(x21*x5 - x25 + x9);
    double dxdudu = x1*(x26 + x32)/pow(f_x, 3);
    double dxdvdu = x35;
    double dydudu = x24*x34;
    double dydvdu = x39;
    double dxdudv = x35;
    double dxdvdv = x11*x3*x38;
    double dydudv = x39;
    double dydvdv = x4*(x26 + x37)/pow(f_y, 3);
    // clang-format on

    std::cout << dxdudv << " " << dxdvdu << std::endl;

    cv::Mat_<double> ret(2, 2, CV_64F);
    ret << dxdu, dxdv, dydu, dydv;
    J = ret;
}

void UndistortPointG(cv::Point2d uv, cv::Point2d& xy, cv::Mat& J,
                     cv::Mat camera_matrix, cv::Mat distortion_cooficients) {
    static constexpr double eps = 1e-9;
    static constexpr int kNumIterations = 9;
    auto& K = static_cast<cv::Mat_<double>&>(camera_matrix);
    auto& D = static_cast<cv::Mat_<double>&>(distortion_cooficients);

    double f_x = K(0, 0), f_y = K(1, 1), c_x = K(0, 2), c_y = K(1, 2);
    double u = uv.x, v = uv.y;

    // clang-format off
    double x0 = pow(f_x, -2);
    double x1 = c_x - u;
    double x2 = x0*pow(x1, 2);
    double x3 = pow(f_y, -2);
    double x4 = c_y - v;
    double x5 = x3*pow(x4, 2);
    double x6 = x2 + x5;
    double x7 = sqrt(x6);
    
    double theta_ = x7;
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

    double x8 = tan(theta);
    double x9 = x8/x7;
    double x10 = 1.0/f_x;
    double x11 = x1*x10;
    double x12 = 1.0/f_y;
    double x13 = x12*x4;
    double x14 = pow(x6, -3.0/2.0);
    double x15 = x14*x8;
    double x16 = x15*x2;
    double x17 = pow(x8, 2) + 1;
    double x18 = theta;
    double x19 = dthetaDtheta_;
    double x20 = x17*x19;
    double x21 = x20/x6;
    double x22 = -x15 + x21;
    double x23 = x3*x4;
    double x24 = x0*x13;
    double x25 = x15*x5;
    double x26 = 3*x15 - 3*x21;
    double x27 = 3*x2;
    double x28 = x8/pow(x6, 5.0/2.0);
    double x29 = x14*x17*dthetaDtheta_Dtheta_;
    double x30 = x20/pow(x6, 2);
    double x31 = 2*x17*pow(x19, 2);
    double x32 = -x16*x31 - x2*x29 - x27*x28 + x27*x30;
    double x33 = x15 - x21;
    double x34 = x32 + x33;
    double x35 = x10*x23*x34;
    double x36 = 3*x5;
    double x37 = -x25*x31 - x28*x36 - x29*x5 + x30*x36;
    double x38 = x33 + x37;
    double x39 = x0*x1*x12*x38;

    double x = -x11*x9;
    double y = -x13*x9;
    double dxdu = x10*(-x16 + x2*x21 + x9);
    double dxdv = x11*x22*x23;
    double dydu = x1*x22*x24;
    double dydv = x12*(x21*x5 - x25 + x9);
    double dxdudu = x1*(x26 + x32)/pow(f_x, 3);
    double dxdvdu = x35;
    double dydudu = x24*x34;
    double dydvdu = x39;
    double dxdudv = x35;
    double dxdvdv = x11*x3*x38;
    double dydudv = x39;
    double dydvdv = x4*(x26 + x37)/pow(f_y, 3);
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
    UndistortPointG(cv::Point2d{2000, 1600}, upoint, J, c.CameraMatrix(),
                    c.DistortionCoeffs());

    std::cout << upoint << std::endl << J * P.at<double>(0, 0) << std::endl;

    auto A = AffineApproximation(c.CameraMatrix(), c.DistortionCoeffs(), P,
                                 cv::Point2f{2000, 1600}, cv::Size(1, 1));

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