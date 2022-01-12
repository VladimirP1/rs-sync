
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

cv::Mat_<double> To4x4(const cv::Mat_<double>& in) {
    cv::Mat_<double> out = cv::Mat::eye(4, 4, CV_64F);
    in.copyTo(out(cv::Rect(cv::Point(0, 0), in.size())));
    return out;
}

bool RoiInSize(cv::Rect roi, cv::Size size) {
    return roi.x >= 0 && roi.y >= 0 && roi.x + roi.width < size.width &&
           roi.y + roi.height < size.height;
}

bool ExtractUndistortedPatch(const cv::Mat& image, cv::Mat& patch, cv::Point2d center, cv::Mat P,
                             cv::Size src_size, cv::Size dst_size) {
    cv::Point2d src_center{src_size.width / 2., src_size.height / 2.};
    cv::Point2d dst_center{dst_size.width / 2., dst_size.height / 2.};

    auto roi = cv::Rect(center - src_center, src_size);
    // std::cout << roi << std::endl;
    if (!RoiInSize(roi, image.size())) {
        return false;
    }

    cv::Mat dpatch = image(roi);

    cv::Mat_<double> T = P.clone();
    cv::Mat_<double> M = cv::Mat::eye(3, 3, CV_64F);
    M.col(2) << (center.x - src_center.x), (center.y - src_center.y), 1.;
    T = T * M;

    cv::Mat_<double> cp(3, 1, CV_64F);
    cp << src_center.x, src_center.y, 1;

    cv::Mat_<double> v = -T * cp;
    v /= v(2);
    M(0, 2) = dst_center.x - v(0);
    M(1, 2) = dst_center.y - v(1);
    T = M * T;

    cv::warpPerspective(dpatch, patch, T(cv::Rect(0, 0, 3, 3)), dst_size, cv::INTER_CUBIC,
                        cv::BORDER_CONSTANT, cv::Scalar(0, 255, 0));
    return true;
}

void MatchPatches(const cv::Mat& prev, const cv::Mat& next, cv::Mat& corr, cv::Size template_size) {
    cv::Mat f_prev, f_templ;
    prev.convertTo(f_prev, CV_32FC3);
    cv::Mat templ = next(cv::Rect((next.cols - template_size.width) / 2.,
                                  (next.rows - template_size.height) / 2., template_size.width,
                                  template_size.height));
    templ.convertTo(f_templ, CV_32FC3);

    cv::matchTemplate(f_prev, f_templ, corr, cv::TM_SQDIFF_NORMED);
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

    tracker.InitCorners(reader.CurGray());

    cv::Mat_<double> P = GetProjectionForUndistort(calibration);

    for (int i = 1; i < 800; ++i) {
        reader.Advance();
        auto [old_c, new_c] = tracker.Track(reader.PrevGray(), reader.CurGray());

        std::vector<cv::Point2f> old_u, new_u;
        cv::fisheye::undistortPoints(old_c, old_u, calibration.CameraMatrix(),
                                     calibration.DistortionCoeffs(), cv::Mat::eye(3, 3, CV_32F));
        cv::fisheye::undistortPoints(new_c, new_u, calibration.CameraMatrix(),
                                     calibration.DistortionCoeffs(), cv::Mat::eye(3, 3, CV_32F));

        constexpr double rs_cooef = .75;
        for (int i = 0; i < old_c.size(); ++i) {
            auto scale = (1 + (new_c[i].y - old_c[i].y) / reader.CurGray().rows * rs_cooef);
            new_u[i].x = (new_u[i].x - old_u[i].x) / scale + old_u[i].x;
            new_u[i].y = (new_u[i].y - old_u[i].y) / scale + old_u[i].y;
        }

        std::vector<uchar> mask;
        if (old_u.size() < 5) continue;
        auto E =
            cv::findEssentialMat(old_u, new_u, 1., cv::Point2d(0, 0), cv::RANSAC, .99, .001, mask);

        cv::Mat_<double> R, t, points4d;
        if (E.rows != 3 || E.cols != 3) continue;
        cv::recoverPose(E, old_u, new_u, cv::Mat::eye(3, 3, CV_64F), R, t, 1000, mask, points4d);

        cv::Mat_<double> P0 = cv::Mat::eye(3, 4, R.type());
        cv::Mat_<double> P1(3, 4, R.type());
        P1(cv::Range::all(), cv::Range(0, 3)) = R * 1.0;
        P1.col(3) = t * 1.0;

        const cv::Size src_patch_size{25, 25};
        const cv::Size dst_patch_size{15, 15};
        const cv::Point2d src_patch_center{src_patch_size.width / 2., src_patch_size.height / 2.};
        const cv::Point2d dst_patch_center{dst_patch_size.width / 2., dst_patch_size.height / 2.};
        cv::Mat corr_mos = cv::Mat::zeros(200, 800, CV_8UC3);
        cv::Mat patch0_mos = cv::Mat::zeros(200, 800, CV_8UC3);
        cv::Mat patch1_mos = cv::Mat::zeros(200, 800, CV_8UC3);
        Mosaic mosaic(patch0_mos, dst_patch_size.width);
        for (int i = 0; i < points4d.cols; ++i) {
            if (!mask[i]) continue;

            cv::Mat_<double> point0m = P0 * points4d.col(i);
            cv::Mat_<double> point1m = P1 * points4d.col(i);
            if (point0m(2) < 0) point0m *= -1;
            if (point1m(2) < 0) point1m *= -1;

            cv::Point3d point0 = cv::Point3d{point0m(0), point0m(1), point0m(2)};
            cv::Point3d point1 = cv::Point3d{point1m(0), point1m(1), point1m(2)};

            cv::Point2d dist0, dist1;
            cv::Mat_<double> affineDistort0, affineDistort1, affineUndistort0, affineUndistort1;
            ProjectPointJacobian(point0, dist0, affineDistort0, calibration.CameraMatrix(),
                                 calibration.DistortionCoeffs());
            ProjectPointJacobian(point1, dist1, affineDistort1, calibration.CameraMatrix(),
                                 calibration.DistortionCoeffs());

            cv::invert(affineDistort0, affineUndistort0);
            cv::invert(affineDistort1, affineUndistort1);

            // cv::circle(img, dist0, 5, cv::Scalar(0, 255, 255), 3);
            // cv::circle(img, dist1, 5, cv::Scalar(0, 255, 0), 3);
            // cv::circle(img, old_c[i], 5, cv::Scalar(255, 255, 255), 3);

            // if (std::abs(dist0.x - 2000) > 1000 || std::abs(dist0.y - 1500) > 1000) continue;

            cv::Mat P0_inv, P1_inv;
            cv::invert(To4x4(P1), P1_inv);
            cv::invert(To4x4(P0), P0_inv);

            cv::Mat upatch0, upatch1, ucorr;
            bool success = true;
            success &= ExtractUndistortedPatch(
                reader.Prev(), upatch0, dist0,
                P * ProjectionTo2d(P0_inv, points4d(2, i), points4d(3, i), 1.) * affineUndistort0,
                src_patch_size, dst_patch_size);

            success &= ExtractUndistortedPatch(
                reader.Cur(), upatch1, dist1,
                P * ProjectionTo2d(P1_inv, points4d(2, i), points4d(3, i), 1.) * affineUndistort1,
                src_patch_size, dst_patch_size);

            if (!success) continue;

            MatchPatches(upatch0, upatch1, ucorr, cv::Size(8, 8));

            cv::Point minloc;
            cv::minMaxLoc(ucorr, nullptr, nullptr, &minloc);
            ucorr.convertTo(ucorr, CV_8UC1, 255);
            cv::cvtColor(ucorr, ucorr, cv::COLOR_GRAY2BGR);
            ucorr.at<cv::Vec3b>(minloc.y, minloc.x) = {255, 255, 0};

            ucorr.at<cv::Vec3b>(ucorr.rows / 2, ucorr.cols / 2) = {0, 255, 255};

            mosaic.Add(patch0_mos, upatch0);
            mosaic.Add(patch1_mos, upatch1);
            mosaic.Add(corr_mos, ucorr);

            mosaic.Advance();
        }

        cv::imwrite("out" + std::to_string(i) + "a.jpg", patch0_mos);
        cv::imwrite("out" + std::to_string(i) + "b.jpg", patch1_mos);
        cv::imwrite("out" + std::to_string(i) + "c.jpg", corr_mos);
    }

    return 0;
}