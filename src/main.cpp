#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <io/stopwatch.hpp>
#include <vision/calibration.hpp>
#include <vision/camera_model.hpp>
#include <vision/utils.hpp>

cv::Mat GetP(const FisheyeCalibration& calibration) {
    cv::Mat P;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        calibration.CameraMatrix(), calibration.DistortionCoeffs(),
        cv::Size(calibration.Width(), calibration.Height()),
        cv::Mat::eye(3, 3, CV_64F), P);
    return P;
}

cv::Point2d MatToPoint31d(cv::Mat_<double> m) {
    return {m(0) / m(2), m(1) / m(2)};
}

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

    std::vector<cv::Point2d> corners;
    cv::goodFeaturesToTrack(gray, corners, 2000, .01, 50);

    std::cout << "Found " << corners.size() << " corners" << std::endl;

    int patch_size = 24;
    int dst_patch_size = 20;
    cv::Mat corr, result;
    cv::Mat corr_mos = cv::Mat::zeros(800, 800, frame.type());
    cv::Mat patch_mos = cv::Mat::zeros(800, 800, frame.type());
    Mosaic mosaic(corr_mos, dst_patch_size * 2);
    cv::Mat_<double> A, PS = P.clone();
    PS(0, 2) = PS(1, 2) = 0.;
    {
        Stopwatch stopwatch("Untistorting patches");
        for (int i = 0; i < corners.size(); ++i) {
            cv::Point2d center = corners[i];

            if (center.x + patch_size > frame.cols ||
                center.y + patch_size > frame.rows ||
                center.x - patch_size < 0 || center.y - patch_size < 0) {
                continue;
            }

            cv::Mat patch =
                frame(cv::Rect(center - cv::Point2d{static_cast<double>(patch_size), static_cast<double>(patch_size)},
                               cv::Size(patch_size * 2, patch_size * 2)));

            cv::Point2d xy;
            UndistortPointJacobian(center, xy, A, c.CameraMatrix(),
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

            const int filter_size = 10;
            result.convertTo(corr, CV_32FC3);
            cv::Mat templ = corr(cv::Rect(
                dst_patch_size - filter_size, dst_patch_size - filter_size,
                2 * filter_size, 2 * filter_size));

            cv::matchTemplate(corr, templ, corr, cv::TM_SQDIFF_NORMED);
            corr.convertTo(corr, CV_8UC1, 255);
            cv::cvtColor(corr, corr, cv::COLOR_GRAY2BGR);

            result(cv::Rect(cv::Point(0, 0), result.size())) = cv::Scalar(127,127,127);
            corr.copyTo(result(cv::Rect(cv::Point((dst_patch_size - filter_size)/2 - 1, (dst_patch_size - filter_size)/2 - 1), corr.size())));

            mosaic.Add(corr_mos, result);
            mosaic.Advance();
        }
    }
    cv::imwrite("out_corr.png", corr_mos);
    cv::imwrite("out_patch.png", patch_mos);
}