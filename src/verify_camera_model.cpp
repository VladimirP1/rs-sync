#include <iostream>

#include <bl/context.hpp>
#include <bl/calibration_provider.hpp>

#include <vision/camera_model.hpp>

#include <Eigen/Eigen>

#include <opencv2/calib3d.hpp>

using namespace rssync;

int main() {
    auto ctx = IContext::CreateContext();
    RegisterCalibrationProvider(ctx, kCalibrationProviderName,
                                "hawkeye_firefly_x_lite_4k_43_v2.json");
    ctx->ContextLoaded();

    auto calibration =
        ctx->GetComponent<ICalibrationProvider>(kCalibrationProviderName)->GetCalibraiton();

    cv::Point3d point{-0.041985, -0.0847903, 1};

    {  // Impl 1
        cv::Point2d dist_point;
        cv::Mat_<double> apx_distort;

        ProjectPointJacobian(point, dist_point, apx_distort, calibration.CameraMatrix(),
                             calibration.DistortionCoeffs());

        std::cout << dist_point.x << " " << dist_point.y << " " << apx_distort(0, 0) << std::endl;
    }

    {  // Impl 2
        Eigen::Vector3d point_e;
        point_e << point.x, point.y, point.z;

        Eigen::Matrix<double, 8, 1> lens_params;
        lens_params << calibration.CameraMatrix()(0, 0), calibration.CameraMatrix()(1, 1),
            calibration.CameraMatrix()(0, 2), calibration.CameraMatrix()(1, 2),
            calibration.DistortionCoeffs()(0), calibration.DistortionCoeffs()(1),
            calibration.DistortionCoeffs()(2), calibration.DistortionCoeffs()(3);
        double uv[2];
        double du[11], dv[11];
        ProjectPointJacobianExtended(point_e.data(), lens_params.data(), uv, du, dv);

        std::cout << uv[0] << " " << uv[1] << " " << du[0] << std::endl;
    }

    {  // OpenCV impl
        cv::Mat_<cv::Vec3d> points(1, 1, CV_64FC3);
        points << cv::Vec3d{point.x, point.y, point.z};

        cv::Mat_<cv::Vec2d> projected;
        cv::Mat jacobian;

        cv::fisheye::projectPoints(points, projected, cv::Mat::zeros(3, 1, CV_64F),
                                   cv::Mat::zeros(3, 1, CV_64F), calibration.CameraMatrix(),
                                   calibration.DistortionCoeffs(), 0, jacobian);

        std::cout << projected(0,0)[0] << " " << projected(0,0)[1] << std::endl;
    }

    return 0;
}