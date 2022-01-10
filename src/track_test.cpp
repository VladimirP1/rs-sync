
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
#include <opencv2/video/tracking.hpp>
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
    VideoReader reader("141101AA.MP4");
    // VideoReader reader("193653AA.MP4");

    TrackerImpl tracker(50, 700);
    FisheyeCalibration calibration;

    std::ifstream("GoPro_Hero6_2160p_43.json") >> calibration;

    // reader.SetPosition(42e3);
    reader.SetPosition(10e3);

    tracker.InitCorners(reader.CurGray());

    auto start = std::chrono::steady_clock::now();

    ////
    cv::Mat P;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        calibration.CameraMatrix(), calibration.DistortionCoeffs(),
        cv::Size(reader.CurGray().cols, reader.CurGray().rows),
        cv::Mat::eye(3, 3, CV_32F), P);

    ////

    cv::Mat img;
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
        cv::Mat R, t, points4d;

        if (E.rows != 3 || E.cols != 3) continue;
        cv::recoverPose(E, old_u, new_u, cv::Mat::eye(3, 3, CV_64F), R, t, 1000,
                        mask, points4d);

        cv::Mat v = (cv::Mat_<double>(3, 1) << 0, 0, 1);

        cv::Mat v1 = R * v;
        v1 /= v1.at<double>(2);

        cv::Mat Rv;
        cv::Rodrigues(R, Rv);
        std::cout << cv::norm(Rv) * 180 / 3.14 << std::endl;

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

        cv::Mat P0 = cv::Mat::eye(3, 4, R.type());
        cv::Mat P1(3, 4, R.type());
        P1(cv::Range::all(), cv::Range(0, 3)) = R * 1.0;
        P1.col(3) = t * 1.0;

        std::vector<cv::Point2d> d_points1;
        std::vector<cv::Point2d> d_points0;
        std::vector<double> depths;
        for (int i = 0; i < points4d.cols; ++i) {
            if (mask[i]) {
                auto Q = points4d.col(i);
                Q.row(0) /= Q.row(3);
                Q.row(1) /= Q.row(3);
                Q.row(2) /= Q.row(3);
                Q.row(3) /= Q.row(3);

                d_points0.push_back({Q.at<double>(0) / Q.at<double>(2),
                                     Q.at<double>(1) / Q.at<double>(2)});
                // depths.push_back(Q.at<double>(2));
                Q = P1 * Q;
                cv::Mat m(3, 1, CV_64F);
                m = Q(cv::Range(0, 3), cv::Range(0, 1)).clone();
                d_points1.push_back({m.at<double>(0) / m.at<double>(2),
                                     m.at<double>(1) / m.at<double>(2)});
                depths.push_back(m.at<double>(2));
            }
        }

        std::vector<cv::Point2d> dd_points1;

        cv::fisheye::distortPoints(d_points1, dd_points1,
                                   calibration.CameraMatrix(),
                                   calibration.DistortionCoeffs());

        std::vector<cv::Point2d> dd_points0;

        cv::fisheye::distortPoints(d_points0, dd_points0,
                                   calibration.CameraMatrix(),
                                   calibration.DistortionCoeffs());

        img = reader.Cur().clone();

        cv::line(img, cv::Point2f(img.cols / 2., img.rows / 2.),
                 cv::Point2f(img.cols / 2. + v1.at<double>(0) * 3000,
                             img.rows / 2. + v1.at<double>(1) * 3000),
                 cv::Scalar(0, 255, 0), 4);

        for (int i = 0; i < old_c.size(); ++i) {
            cv::line(img, old_c[i], new_c[i],
                     cv::Scalar(0, mask[i] ? 255 : 0, 255), 3);
        }

        for (int i = 0; i < dd_points1.size(); ++i) {
            cv::circle(img, dd_points1[i], 5, cv::Scalar(0, depths[i], 255), 3);

            cv::line(img, dd_points1[i], dd_points0[i], cv::Scalar(255, 0, 255),
                     3);
        }


        // extract patches around features
        const int patch_size = 12;
        const int half_size = patch_size / 2;
        cv::Mat patches0 = cv::Mat::zeros(1000, 1000, CV_8UC3);
        cv::Mat patches1 = cv::Mat::zeros(1000, 1000, CV_8UC3);
        {
            double rms = 0;
            int i{0}, j{0};
            for (int t = 0; t < dd_points0.size(); ++t) {
                auto p0 = dd_points0[t];
                auto p1 = dd_points1[t];
                if (p0.y + half_size < img.rows &&
                    p0.x + half_size < img.cols &&
                    p1.y + half_size < img.rows &&
                    p1.x + half_size < img.cols && p0.y - half_size > 0 &&
                    p0.x - half_size > 0 && p1.y - half_size > 0 &&
                    p1.x - half_size > 0) {
                    std::cout << i << " " << j << " " << p0.x << " " << p0.y
                              << " " << p1.x << " " << p1.y << std::endl;

                    cv::Mat dst0(
                        patches0(cv::Rect(i, j, patch_size, patch_size)));
                    cv::Mat dst1(
                        patches1(cv::Rect(i, j, patch_size, patch_size)));
                    reader
                        .Prev()(cv::Rect(p0.x - half_size, p0.y - half_size,
                                         patch_size, patch_size))
                        .copyTo(dst0);

                    reader
                        .Cur()(cv::Rect(p1.x - half_size, p1.y - half_size,
                                        patch_size, patch_size))
                        .copyTo(dst1);
                    std::vector<cv::Point2f> pts0, pts1;
                    std::vector<uchar> status;
                    std::vector<float> err;
                    pts0.push_back({half_size, half_size});
                    cv::calcOpticalFlowPyrLK(dst0, dst1, pts0, pts1, status,
                                             err);

                    auto diff = pts0[0] - pts1[0];
                    rms += diff.dot(diff);
                    // rms += std::sqrt(diff.dot(diff));

                    j += patch_size;
                    if (j + patch_size >= patches0.cols) {
                        j = 0;
                        i += patch_size;
                    }
                }
                if (i + patch_size >= patches0.rows) {
                    break;
                }
            }
            rms = std::sqrt(rms / dd_points0.size());
            // rms = rms / dd_points0.size();
            cv::putText(patches0, std::to_string(rms), cv::Point(500, 900),
                        cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(patches1, std::to_string(rms), cv::Point(500, 900),
                        cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::line(img, cv::Point2f(15,400), cv::Point2f(15,400 - rms * 50), cv::Scalar(255,0,0), 20);
        }
        // cv::imwrite("out" + std::to_string(i) + "a.jpg", patches0);
        // cv::imwrite("out" + std::to_string(i) + "b.jpg", patches1);
        cv::imwrite("out" + std::to_string(i) + ".jpg", img);

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