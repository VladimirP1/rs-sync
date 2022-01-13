
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

    cv::Point2d actual_src_center =
        center - cv::Point2d{static_cast<double>(roi.x), static_cast<double>(roi.y)};

    // std::cout << roi << std::endl;
    if (!RoiInSize(roi, image.size())) {
        return false;
    }

    cv::Mat dpatch = image(roi);

    cv::Mat_<double> T = P.clone();
    cv::Mat_<double> M = cv::Mat::eye(3, 3, CV_64F);
    M.col(2) << (center.x - actual_src_center.x), (center.y - actual_src_center.y), 1.;
    T = T * M;

    cv::Mat_<double> cp(3, 1, CV_64F);
    cp << src_center.x, src_center.y, 1;
    cv::Mat_<double> v = -T * cp;

    M.col(2) << dst_center.x - v(0) / v(2), dst_center.y - v(1) / v(2), 1;
    T = M * T;

    cv::warpPerspective(dpatch, patch, T(cv::Rect(0, 0, 3, 3)), dst_size, cv::INTER_CUBIC,
                        cv::BORDER_CONSTANT, cv::Scalar(0, 255, 0));
    return true;
}

void MatchPatches(const cv::Mat& prev, const cv::Mat& next, cv::Mat& corr, double margin) {
    cv::Mat f_prev, f_templ;
    prev.convertTo(f_prev, CV_32FC3);
    cv::Mat templ = next(cv::Rect(margin, margin, next.cols - 2 * margin, next.rows - 2 * margin));
    templ.convertTo(f_templ, CV_32FC3);

    cv::matchTemplate(f_prev, f_templ, corr, cv::TM_SQDIFF_NORMED);
}

cv::Point2d MinSubpixel(cv::Mat_<float> img) {
    double min;
    cv::Mat thresh;
    cv::minMaxLoc(img, &min);
    cv::threshold(img, thresh, min * 2, 1., cv::THRESH_BINARY_INV);
    cv::Moments m = cv::moments(thresh, true);
    return {m.m10 / m.m00 + .5, m.m01 / m.m00 + .5};
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

    reader.SetPosition(120e3);

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
        cv::recoverPose(E, old_u, new_u, cv::Mat::eye(3, 3, CV_64F), R, t, 100000, mask, points4d);

        cv::Mat_<double> P0 = cv::Mat::eye(3, 4, R.type());
        cv::Mat_<double> P1(3, 4, R.type());
        P1(cv::Range::all(), cv::Range(0, 3)) = R * 1.0;
        P1.col(3) = t * 1.0;

        cv::Point2d shift{0, 0};
        double sum = 0;
        int processed = 0;
        cv::Mat sum_ucorr;
        const cv::Size src_patch_size{25, 25};
        const cv::Size dst_patch_size{15, 15};
        const cv::Point2d src_patch_center{src_patch_size.width / 2., src_patch_size.height / 2.};
        const cv::Point2d dst_patch_center{dst_patch_size.width / 2., dst_patch_size.height / 2.};
        cv::Mat corr_mos = cv::Mat::zeros(800, 800, CV_8UC3);
        cv::Mat patch0_mos = cv::Mat::zeros(200, 800, CV_8UC3);
        cv::Mat patch1_mos = cv::Mat::zeros(200, 800, CV_8UC3);
        Mosaic mosaic(patch0_mos, dst_patch_size.width);
        Mosaic mosaic2(corr_mos, dst_patch_size.width * 4);
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

            MatchPatches(upatch0, upatch1, ucorr, 4);

            // ucorr = cv::Mat::ones(ucorr.size(), CV_32FC1);
            // ucorr.at<float>(5, 5) = 0;
            cv::Point2d minloc = MinSubpixel(ucorr);
            shift += minloc;
            ++processed;

            // cv::Mat xucorr, yucorr;
            // cv::Sobel(ucorr, xucorr, CV_32F, 1, 0);
            // cv::Sobel(ucorr, yucorr, CV_32F, 0, 1);
            // ucorr = xucorr * xucorr + yucorr * yucorr;

            double min, max;
            cv::minMaxLoc(ucorr, &min, &max);
            sum += min;
            ucorr -= min;
            ucorr /= (max - min);

            if (sum_ucorr.cols == 0) {
                sum_ucorr = ucorr.clone();
            } else {
                sum_ucorr += ucorr;
            }

            ucorr.convertTo(ucorr, CV_8UC1, 255);

            cv::cvtColor(ucorr, ucorr, cv::COLOR_GRAY2BGR);

            cv::resize(ucorr, ucorr, ucorr.size() * 6, 0, 0, cv::INTER_LINEAR);

            cv::applyColorMap(ucorr, ucorr, cv::COLORMAP_MAGMA);

            cv::circle(ucorr, {ucorr.cols / 2, ucorr.rows / 2}, 1, cv::Scalar(0, 255, 0), 1,
                       cv::LINE_AA);
            // cv::circle(ucorr, minloc * 6, 2, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

            mosaic.Add(patch0_mos, upatch0);
            mosaic.Add(patch1_mos, upatch1);
            mosaic2.Add(corr_mos, ucorr);

            mosaic.Advance();
            mosaic2.Advance();
        }

        std::cout << sum / processed << shift / processed << std::endl;

        sum_ucorr /= processed;
        sum_ucorr.convertTo(sum_ucorr, CV_8UC1, 255);
        cv::cvtColor(sum_ucorr, sum_ucorr, cv::COLOR_GRAY2BGR);
        cv::resize(sum_ucorr, sum_ucorr, sum_ucorr.size() * 6, 0, 0, cv::INTER_LINEAR);
        cv::applyColorMap(sum_ucorr, sum_ucorr, cv::COLORMAP_MAGMA);
        cv::circle(sum_ucorr, {sum_ucorr.cols / 2, sum_ucorr.rows / 2}, 1, cv::Scalar(0, 255, 0), 1,
                   cv::LINE_AA);
        mosaic2.Add(corr_mos, sum_ucorr);
        mosaic2.Advance();
        cv::resize(sum_ucorr,sum_ucorr,sum_ucorr.size()*4);
        cv::Mat imgz = reader.Cur().clone();
        sum_ucorr.copyTo(imgz(cv::Rect(cv::Point(0,0), sum_ucorr.size())));
        sum_ucorr = cv::Mat{};

        cv::imwrite("out" + std::to_string(i) + "a.jpg", patch0_mos);
        cv::imwrite("out" + std::to_string(i) + "b.jpg", patch1_mos);
        cv::imwrite("out" + std::to_string(i) + "c.jpg", corr_mos);
        cv::imwrite("out" + std::to_string(i) + ".jpg", imgz);
    }

    return 0;
}