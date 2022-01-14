#include "camera_model.hpp"

void UndistortPointJacobian(cv::Point2d uv, cv::Point2d& xy, cv::Mat& A,
                            cv::Mat camera_matrix,
                            cv::Mat distortion_cooficients) {
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

    cv::Mat_<double> t(3, 3, CV_64F);
    t << 
        1., 0., xy.x - uv.x * dxdu - uv.y * dxdv, 
        0., 1., xy.y - uv.x * dydu - uv.y * dydv,
        0., 0., 1.;
    // clang-format on

    A = t * ret;
}

void ProjectPointJacobian(cv::Point3d xyz, cv::Point2d& uv, cv::Mat& A,
                          cv::Mat camera_matrix,
                          cv::Mat distortion_cooficients) {
    auto& K = static_cast<cv::Mat_<double>&>(camera_matrix);
    auto& D = static_cast<cv::Mat_<double>&>(distortion_cooficients);

    double a = xyz.x / xyz.z;
    double b = xyz.y / xyz.z;
    double dadx = 1. / xyz.z;
    double dbdy = 1. / xyz.z;

    double r = std::sqrt(a * a + b * b);
    double drda = a / r;
    double drdb = b / r;

    double theta = std::atan(r);
    double dthetaDr = 1. / (1 + r * r);

    double theta2 = theta * theta, theta3 = theta2 * theta,
           theta4 = theta2 * theta2, theta5 = theta2 * theta3,
           theta6 = theta3 * theta3, theta7 = theta3 * theta4,
           theta8 = theta4 * theta4, theta9 = theta4 * theta5;
    double theta_ =
        theta + D(0) * theta3 + D(1) * theta5 + D(2) * theta7 + D(3) * theta9;
    double dtheta_dtheta = 1 + 3 * D(0) * theta2 + 5 * D(1) * theta4 +
                           7 * D(2) * theta6 + 8 * D(3) * theta8;

    double k = theta_ / r;
    double dkdr = (dtheta_dtheta * dthetaDr * r - theta_) / (r * r);
    // double dkdtheta = (dtheta_dtheta * r - theta_ / dthetaDr) / (r * r);

    double x_ = a * k, y_ = b * k;
    double dx_dx = dadx * k + (dkdr * drda * dadx) * a;
    double dy_dy = dbdy * k + (dkdr * drdb * dbdy) * b;
    double dx_dy = (dkdr * drdb * dbdy) * a;
    double dy_dx = (dkdr * drda * dadx) * b;

    double u = K(0, 0) * x_ + K(0, 2), v = K(1, 1) * y_ + K(1, 2);
    double dudx_ = K(0, 0);
    double dvdy_ = K(1, 1);

    double dudx = dudx_ * dx_dx;
    double dudy = dudx_ * dx_dy;
    double dvdx = dvdy_ * dy_dx;
    double dvdy = dvdy_ * dy_dy;

    uv = {u, v};

    // clang-format off
    cv::Mat_<double> ret(3, 3, CV_64F);
    ret << 
        dudx, dudy, 0., 
        dvdx, dvdy, 0.,
        0.,   0.,   1.;

    cv::Mat_<double> t(3, 3, CV_64F);
    t << 
        1., 0., (uv.x - a * dudx - b * dudy), 
        0., 1., (uv.y - a * dvdx - b * dvdy),
        0., 0., 1;
    // clang-format on

    A = t * ret;
}