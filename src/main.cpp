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

void UndistortPointG(cv::Point2d uv, cv::Point2d& xy, cv::Mat& A,
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

    double theta = M_PI / 4.;
    double dthetaDtheta_ = 0;
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
        dthetaDtheta_ = 1. / cur_dTheta_;
        double new_theta = theta - error * dthetaDtheta_;
        while (new_theta >= M_PI / 2. || new_theta <= 0.) {
            new_theta = (new_theta + theta) / 2.;
        }
        theta = new_theta;
    }

    double r = std::tan(theta);
    double inv_cos_theta = 1. / std::cos(theta);
    double drDtheta = inv_cos_theta * inv_cos_theta;

    double s = (theta_ < eps) ? inv_cos_theta : r / theta_;
    double drDtheta_ = drDtheta * dthetaDtheta_;
    double dsDtheta_ =
        (theta_ < eps) ? 0. : (drDtheta_ * theta_ - r * 1) / theta_ / theta_;

    double dxdu = dx_du * s + x_ * dsDtheta_ * dtheta_dx_ * dx_du;
    double dydv = dy_dv * s + y_ * dsDtheta_ * dtheta_dy_ * dy_dv;
    double dxdv = x_ * dsDtheta_ * dtheta_dy_ * dy_dv;
    double dydu = y_ * dsDtheta_ * dtheta_dx_ * dx_du;

    xy = {x_ * s, y_ * s};

    // clang-format off
    cv::Mat_<double> ret(3, 3, CV_64F);
    ret << 
        dxdu, dxdv, 0., 
        dydu, dydv, 0.,
        0.,   0.,   1.;
    // clang-format on
    A = ret;
}

cv::Point2d MatToPoint31d(cv::Mat_<double> m) {
    return {m(0) / m(2), m(1) / m(2)};
}

struct Tim {
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    ~Tim() {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> dur = (end - start);
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(dur)
                         .count()
                  << std::endl;
    }
};

class Mosaic {
   public:
    Mosaic(const cv::Mat& output, int patch_size)
        : rows_(output.rows), cols_(output.cols), patch_size_(patch_size) {}

    bool Add(cv::Mat& out, const cv::Mat& m) {
        if (j + patch_size_ >= cols_ || i + patch_size_ >= rows_) {
            return false;
        }

        m.copyTo(out(cv::Rect(j, i, patch_size_, patch_size_)));

        return true;
    }

    bool Advance() {
        j += patch_size_;
        if (j + patch_size_ >= cols_) {
            j = 0;
            i += patch_size_;
        }

        if (i + patch_size_ >= rows_) {
            return false;
        }
        return true;
    }

   private:
    int i{}, j{};
    int rows_, cols_;
    int patch_size_;
};

int main(int argc, char** argv) {
    std::ifstream fs("GoPro_Hero6_2160p_43.json");

    FisheyeCalibration c;
    fs >> c;
    auto P = GetP(c);

    std::vector<cv::Point2d> upts, pts{cv::Point2d{2009, 1507}};
    cv::fisheye::undistortPoints(pts, upts, c.CameraMatrix(),
                                 c.DistortionCoeffs());
    std::cout << upts[0] << std::endl;

    cv::VideoCapture cap("141101AA.MP4");

    cv::Mat frame, gray;
    cap.set(cv::CAP_PROP_POS_MSEC, 42e3);
    cap.read(frame);
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // cv::Mat undistorted;
    // {
    //     Tim t;
    //     cv::fisheye::undistortImage(frame, undistorted, c.CameraMatrix(),
    //                                 c.DistortionCoeffs(), P);
    // }
    // cv::imwrite("outd.png", frame);
    // cv::imwrite("outu.png", undistorted);

    std::vector<cv::Point2d> corners;
    cv::goodFeaturesToTrack(gray, corners, 2000, .01, 50);

    std::cout << "Found " << corners.size() << " corners" << std::endl;

    double patch_size = 15;
    double dst_patch_size = 10;
    cv::Mat corr, result;
    cv::Mat corr_mos = cv::Mat::zeros(800, 800, frame.type());
    cv::Mat patch_mos = cv::Mat::zeros(800, 800, frame.type());
    Mosaic mosaic(corr_mos, dst_patch_size * 2);
    cv::Mat_<double> A, PS = P.clone();
    PS(0, 2) = PS(1, 2) = 0.;
    {
        Tim t;
        for (int i = 0; i < corners.size(); ++i) {
            cv::Point2d center = corners[i];

            if (center.x + patch_size > frame.cols ||
                center.y + patch_size > frame.rows ||
                center.x - patch_size < 0 || center.y - patch_size < 0) {
                continue;
            }

            cv::Mat patch =
                frame(cv::Rect(center - cv::Point2d{patch_size, patch_size},
                               cv::Size(patch_size * 2, patch_size * 2)));

            cv::Point2d xy;
            UndistortPointG(center, xy, A, c.CameraMatrix(),
                            c.DistortionCoeffs());

            cv::Mat_<double> T = (PS * A);

            cv::Mat_<double> cp(3, 1, CV_64F);
            cp << patch_size, patch_size, 1;

            T.col(2) = -T * cp;
            T(0, 2) += dst_patch_size;
            T(1, 2) += dst_patch_size;

            cv::warpAffine(patch, result, T(cv::Rect(0, 0, 3, 2)),
                           cv::Size(dst_patch_size * 2, dst_patch_size * 2),
                           cv::INTER_CUBIC, cv::BORDER_CONSTANT,
                           cv::Scalar(0, 255, 0));

            mosaic.Add(patch_mos, result);

            const double filter_size = 4;
            result.convertTo(corr, CV_32FC3);
            cv::Mat templ = corr(cv::Rect(
                dst_patch_size - filter_size, dst_patch_size - filter_size,
                2 * filter_size, 2 * filter_size));

            cv::matchTemplate(corr, templ, corr, cv::TM_SQDIFF_NORMED);
            corr.convertTo(corr, CV_8UC1, 255);
            cv::cvtColor(corr, corr, cv::COLOR_GRAY2BGR);

            result(cv::Rect(cv::Point(0, 0), result.size())) = cv::Scalar(0,0,0);
            corr.copyTo(result(cv::Rect(cv::Point(0, 0), corr.size())));

            mosaic.Add(corr_mos, result);
            mosaic.Advance();
        }
    }
    cv::imwrite("out_corr.png", corr_mos);
    cv::imwrite("out_patch.png", patch_mos);
}