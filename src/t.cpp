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

int main() {
    VideoReader reader("141101AA.MP4");
    reader.SetPosition(42e3);

    FisheyeCalibration calibration;
    std::ifstream("GoPro_Hero6_2160p_43.json") >> calibration;

    cv::Mat_<double> P = GetProjectionForUndistort(calibration);
    P(0, 0) /= 2;
    P(1, 1) /= 2;

    cv::Mat ud;

    cv::fisheye::undistortImage(reader.Cur(),ud , calibration.CameraMatrix(), calibration.DistortionCoeffs(), P);

    cv::imwrite("out.jpg", ud);

    for (int i = 0; i < 1000; i += 2) {
        cv::Mat_<double> A;
        cv::Point2d xy;
        cv::Point2d uv{i * 4. + 2 * sin(i/10.) * 3., i * 3. - 2 * sin(i/10.) * 4.};
        UndistortPointJacobian(uv, xy, A, calibration.CameraMatrix(),
                               calibration.DistortionCoeffs());

        std::cout << P * A << xy << std::endl;
        cv::Mat in = reader.Cur().clone(), out;

        cv::circle(in, uv, 15, cv::Scalar(0, 0, 255), -1);

        cv::warpPerspective(in, out, P * A, reader.Cur().size());

        cv::imwrite("out" + std::to_string(i) + ".jpg", out);
        cv::imwrite("out" + std::to_string(i + 1) + ".jpg", ud);

    }

    return 0;
}