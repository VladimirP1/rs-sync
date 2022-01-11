
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <vision/calibration.hpp>
#include <vision/tracking.hpp>
#include <vision/video_reader.hpp>
#include <vision/camera_model.hpp>
#include <vision/utils.hpp>

cv::Mat_<double> To3d(const cv::Mat_<double>& in) {
    cv::Mat_<double> out = cv::Mat::eye(3, 4, CV_64F);
    in.col(0).copyTo(out.col(0));
    in.col(1).copyTo(out.col(1));
    in.col(2).copyTo(out.col(3));
    out(2,3) = 0;

    return out;
}

cv::Mat_<double> To2d(const cv::Mat_<double>& in) {
    cv::Mat_<double> out = cv::Mat::zeros(3, 3, CV_64F);
    in.col(0).copyTo(out.col(0));
    in.col(1).copyTo(out.col(1));
    in.col(3).copyTo(out.col(2));
    return out;
}

cv::Mat_<double> To4x4(const cv::Mat_<double>& in) {
    cv::Mat_<double> out = cv::Mat::eye(4, 4, CV_64F);
    in.copyTo(out(cv::Rect(cv::Point(0, 0), in.size())));
    return out;
}

bool RoiInSize(cv::Rect roi, cv::Size size) {
    return roi.x >= 0 && roi.y >= 0 && roi.x + roi.width < size.width &&
           roi.y + roi.height < size.height;
}

int main() {
    // VideoReader reader("GX019642.MP4");
    // VideoReader reader("VID_20220110_135451.mp4");
    VideoReader reader("141101AA.MP4");
    // VideoReader reader("193653AA.MP4");

    TrackerImpl tracker(50, 700);
    FisheyeCalibration calibration;

    // std::ifstream("Google_Pixel_5_Filmic_pro_ultrawide_4k.json") >>
    // calibration;
    std::ifstream("GoPro_Hero6_2160p_43.json") >> calibration;
    // std::ifstream("GoPro_Hero6_2160p_16by9_wide.json") >> calibration;

    reader.SetPosition(42e3);
    // reader.SetPosition(65e3);
    // reader.Advance();

    tracker.InitCorners(reader.CurGray());

    cv::Mat_<double> P = GetProjectionForUndistort(calibration), PZ = P.clone();
    PZ(0, 2) = PZ(1, 2) = 0.;

    std::cout << P << std::endl;

    for (int i = 1; i < 800; ++i) {
        reader.Advance();
        auto [old_c, new_c] =
            tracker.Track(reader.PrevGray(), reader.CurGray());

        std::vector<cv::Point2f> old_u, new_u;
        cv::fisheye::undistortPoints(old_c, old_u, calibration.CameraMatrix(),
                                     calibration.DistortionCoeffs(),
                                     cv::Mat::eye(3, 3, CV_32F));
        cv::fisheye::undistortPoints(new_c, new_u, calibration.CameraMatrix(),
                                     calibration.DistortionCoeffs(),
                                     cv::Mat::eye(3, 3, CV_32F));

        constexpr double rs_cooef = .75;
        for (int i = 0; i < old_c.size(); ++i) {
            auto scale = (1 + (new_c[i].y - old_c[i].y) /
                                  reader.CurGray().rows * rs_cooef);
            new_u[i].x = (new_u[i].x - old_u[i].x) / scale + old_u[i].x;
            new_u[i].y = (new_u[i].y - old_u[i].y) / scale + old_u[i].y;
        }

        std::vector<uchar> mask;
        if (old_u.size() < 5) continue;
        auto E = cv::findEssentialMat(old_u, new_u, 1., cv::Point2d(0, 0),
                                      cv::RANSAC, .99, .001, mask);

        cv::Mat_<double> R, t, points4d;
        if (E.rows != 3 || E.cols != 3) continue;
        cv::recoverPose(E, old_u, new_u, cv::Mat::eye(3, 3, CV_64F), R, t, 1000,
                        mask, points4d);

        cv::Mat_<double> P0 = cv::Mat::eye(3, 4, R.type());
        cv::Mat_<double> P1(3, 4, R.type());
        P1(cv::Range::all(), cv::Range(0, 3)) = R * 1.0;
        P1.col(3) = t * 1.0;

        cv::Mat_<double> points3d_0 = P0 * points4d;
        cv::Mat_<double> points3d_1 = P1 * points4d;

        cv::Mat points2d_0, points2d_1;
        cv::projectPoints(points3d_0, cv::Mat::zeros(3, 1, CV_64F),
                          cv::Mat::zeros(3, 1, CV_64F),
                          calibration.CameraMatrix(),
                          calibration.DistortionCoeffs(), points2d_0);

        cv::projectPoints(points3d_1, cv::Mat::zeros(3, 1, CV_64F),
                          cv::Mat::zeros(3, 1, CV_64F),
                          calibration.CameraMatrix(),
                          calibration.DistortionCoeffs(), points2d_1);

        cv::Mat pimg = reader.Prev(), img;
        cv::fisheye::undistortImage(pimg, img, calibration.CameraMatrix(),
                                            calibration.DistortionCoeffs(), P, cv::Size(4000,3000));

        const cv::Size src_patch_size{30, 30};
        const cv::Point2d src_patch_center{src_patch_size.width / 2.,
                                           src_patch_size.height / 2.};
        for (int i = 0; i < points2d_0.rows; ++i) {
            if (!mask[i]) continue;
            auto p0v = points2d_0.at<cv::Vec2d>(i);
            auto p1v = points2d_1.at<cv::Vec2d>(i);
            auto p0 = cv::Point2d{p0v[0], p0v[1]};
            auto p1 = cv::Point2d{p1v[0], p1v[1]};

            cv::Mat_<double> A;
            cv::Point2d xy;
            UndistortPointJacobian(p0, xy, A, calibration.CameraMatrix(),
                                   calibration.DistortionCoeffs());

            cv::Mat_<double> v(4,1,CV_64F);
            v << p0.x, p0.y, 1, 1;
            std::cout << xy << To4x4(To3d(A)) * v << std::endl;

            auto L = To4x4(P) * To4x4(P0) * To4x4(To3d(A));
            cv::Mat_<double> pp = L * v;
            pp.row(0) /= pp.row(3);
            pp.row(1) /= pp.row(3);
            pp.row(2) /= pp.row(3);
            pp.row(3) /= pp.row(3);
            pp.row(1) /= pp.row(2);
            pp.row(0) /= pp.row(2);
            pp.row(2) /= pp.row(2);
            std::cout << xy << pp << std::endl;
            // std::cout << p0 << " " << pp << std::endl;
            // std::cout << To4x4(To3d(A)) << std::endl;

            auto roi = cv::Rect(p0 - src_patch_center, src_patch_size);

            cv::circle(img, cv::Point2d(pp(0), pp(1)), 5,
                       cv::Scalar(0, 255, 255), 3);
            cv::circle(img, cv::Point2d(old_u[i].x * P(0,0) + P(0,2), old_u[i].y * P(1,1) + P(1,2)), 5,
                       cv::Scalar(255, 255, 0), 3);

        }

        // cv::Mat img = reader.Prev();
        // for (int i = 0; i < points2d_0.rows; ++i) {
        //     if (!mask[i]) continue;
        //     auto p0 = points2d_0.at<cv::Vec2d>(i);
        //     auto p1 = points2d_1.at<cv::Vec2d>(i);

        //     cv::circle(img, cv::Point2d(p0[0], p0[1]), 5,
        //                cv::Scalar(0, 255, 255), 3);
        //     cv::circle(img, cv::Point2d(p1[0], p1[1]), 5,
        //                cv::Scalar(255, 255, 0), 3);
        // }
        cv::imwrite("out" + std::to_string(i) + "a.jpg", img);
        break;
    }

    return 0;
}