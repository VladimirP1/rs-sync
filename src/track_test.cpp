
#include <opencv2/core/hal/interface.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "calibration.hpp"
#include "tracking.hpp"

class VideoReader {
   public:
    VideoReader(std::string filename) : cap_{filename} {}

    void SetPosition(double ms) {
        cap_.set(cv::CAP_PROP_POS_MSEC, ms);
        Advance();
    }

    bool Advance() {
        prev_gray_ = std::move(cur_gray_);
        prev_frame_ = std::move(cur_frame_);
        bool success = cap_.read(cur_frame_);
        cv::cvtColor(cur_frame_, cur_gray_, cv::COLOR_BGR2GRAY);
        return success;
    }

    const cv::Mat& Prev() const { return prev_frame_; }

    const cv::Mat& Cur() const { return cur_frame_; }

    const cv::Mat& PrevGray() const { return prev_gray_; }

    const cv::Mat& CurGray() const { return cur_gray_; }

   private:
    cv::Mat cur_gray_;
    cv::Mat prev_gray_;
    cv::Mat cur_frame_;
    cv::Mat prev_frame_;
    cv::VideoCapture cap_;
};

int main() {
    cv::VideoCapture cap("141101AA.MP4");
    // cv::VideoCapture cap("193653AA.MP4");

    cap.set(cv::CAP_PROP_POS_MSEC, 50e3);

    VideoReader reader("141101AA.MP4");
    TrackerImpl tracker(100, 700);
    FisheyeCalibration calibration;

    std::ifstream("GoPro_Hero6_2160p_43.json") >> calibration;

    reader.SetPosition(42e3);

    tracker.InitCorners(reader.CurGray());

    auto start = std::chrono::steady_clock::now();

    ////
    cv::Mat P;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        calibration.CameraMatrix(), calibration.DistortionCoeffs(),
        cv::Size(4000, 3000), cv::Mat::eye(3, 3, CV_32F), P);

    ////

    cv::Mat img;
    for (int i = 1; i < 300; ++i) {
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

        for (int i = 0; i < old_c.size(); ++i) {
            auto scale = (1 + (new_c[i].y - old_c[i].y) / 4000. * .75);
            new_u[i].x = (new_u[i].x - old_u[i].x) * scale + old_u[i].x;
            new_u[i].y = (new_u[i].y - old_u[i].y) * scale + old_u[i].y;
        }

        std::vector<uchar> mask;
        auto E = cv::findEssentialMat(old_u, new_u, 1., cv::Point2d(0, 0),
                                      cv::RANSAC, .999, .01, mask);
        cv::Mat R, t, points4d;

        cv::recoverPose(E, old_u, new_u, cv::Mat::eye(3, 3, CV_64F), R, t, 1000,
                        mask, points4d);

        cv::Mat v = (cv::Mat_<double>(3, 1) << 0, 0, 1);

        cv::Mat v1 = R * v;
        v1 /= v1.at<double>(2);

        // cv::Mat out;
        // cv::fisheye::undistortImage(img, out, calibration.CameraMatrix(),
        //                             calibration.DistortionCoeffs(), P,
        //                             cv::Size(4000, 3000));

        // for (int i = 0; i < old_c.size(); ++i) {
        //     cv::line(out, old_u[i], new_u[i], cv::Scalar(0, mask[i]?255:0,
        //     255), 3);
        // }

        // cv::line(out, cv::Point2f(2000, 1500), cv::Point2f(2000 +
        // v1.at<double>(0) * 300000, 1500 + v1.at<double>(1) * 300000),
        // cv::Scalar(0,255,0), 4);
        // cv::line(out, cv::Point2f(2000, 1500), cv::Point2f(2000 +
        // v2.at<double>(0) * 300000, 1500 + v2.at<double>(1) * 300000),
        // cv::Scalar(255,255,0), 4); cv::line(out, cv::Point2f(2000, 1500),
        // cv::Point2f(2000 + t.at<double>(0) / (1+t.at<double>(2)) * 300, 1500
        // + t.at<double>(1) /  (1+t.at<double>(2)) * 300),
        // cv::Scalar(255,255,255), 4);

        // cv::imwrite("outz" + std::to_string(i) + ".jpg", out);

        std::vector<cv::Point2d> d_points;
        std::vector<double> depths;
        for (int i = 0; i < points4d.cols; ++i) {
            if (mask[i]) {
                auto Q = points4d.col(i);
                Q.row(0) /= Q.row(3);
                Q.row(1) /= Q.row(3);
                Q.row(2) /= Q.row(3);
                Q.row(3) /= Q.row(3);
                Q.row(0) /= Q.row(2);
                Q.row(1) /= Q.row(2);
                Q.row(3) /= Q.row(2);
                d_points.push_back({Q.at<double>(0), Q.at<double>(1)});
                depths.push_back(Q.at<double>(2));
            }
        }

        std::vector<cv::Point2d> dd_points;

        cv::fisheye::distortPoints(d_points, dd_points,
                                   calibration.CameraMatrix(),
                                   calibration.DistortionCoeffs());

        // img = reader.Cur().clone();

        // cv::line(img, cv::Point2f(2000, 1500),
        //          cv::Point2f(2000 + v1.at<double>(0) * 3000,
        //                      1500 + v1.at<double>(1) * 3000),
        //          cv::Scalar(0, 255, 0), 4);

        // for (int i = 0; i < old_c.size(); ++i) {
        //     cv::line(img, old_c[i], new_c[i],
        //              cv::Scalar(0, mask[i] ? 255 : 0, 255), 3);
        // }

        // for (int i = 0; i < dd_points.size(); ++i) {
        //     cv::circle(img, dd_points[i], 5, cv::Scalar(0, depths[i], 255), 3);
        // }

        // cv::imwrite("outz" + std::to_string(i) + ".jpg", img);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dur = (end - start);
    std::cout
        << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
        << std::endl;

    std::vector<cv::Point2f> old_c, old_u;
    old_c.push_back({4000, 3000});
    cv::fisheye::undistortPoints(old_c, old_u, calibration.CameraMatrix(),
                                 calibration.DistortionCoeffs());
    std::cout << old_u[0] << std::endl;

    img = reader.Cur();
    cv::Mat out;
    cv::fisheye::undistortImage(img, out, calibration.CameraMatrix(),
                                calibration.DistortionCoeffs(), P,
                                cv::Size(4000, 3000));

    std::cout << P << std::endl;

    cv::imwrite("out.png", out);

    return 0;
}