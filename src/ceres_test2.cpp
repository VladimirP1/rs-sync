#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
using ceres::AngleAxisRotatePoint;
using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

struct ExponentialResidual {
    ExponentialResidual(double x, double y, double z) : x_(x), y_(y), z_(z) {}
    template <typename T>
    bool operator()(const T* const center, const T* sigma, const T* n, const T* alpha,
                    T* residual) const {
        T angle_axis[3];
        angle_axis[0] = {};
        angle_axis[1] = {};
        angle_axis[2] = T(*alpha);

        T point_shifted[3];
        point_shifted[0] = x_ - center[0];
        point_shifted[1] = y_ - center[1];
        point_shifted[2] = {};

        T point_rotated[3];
        AngleAxisRotatePoint(angle_axis, point_shifted, point_rotated);
        // residual[0] = ampl[0] / 2. / M_PI / sigma[0] / sigma[1] *
        //               exp(-point_rotated[0] * point_rotated[0] / 2. / (sigma[0] * sigma[0])) *
        //               exp(-point_rotated[1] * point_rotated[1] / 2. / (sigma[1] * sigma[1]));

        residual[0] = sigma[0] * point_rotated[0] * point_rotated[0] +
                      sigma[1] * point_rotated[1] * point_rotated[1];
        residual[0] += n[0] * x_ + n[1] * y_ + n[2];
        residual[0] -= z_;

        T d = point_rotated[0] * point_rotated[0] + point_rotated[1] * point_rotated[1] / 4.;
        residual[0] *= 1. - 1. / (1. + exp(-d * d)); return true;
    }

   private:
    const double x_;
    const double y_;
    const double z_;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    cv::Point max_loc;
    cv::Mat data = cv::imread(argv[1]);
    cv::cvtColor(data, data, cv::COLOR_BGR2GRAY);
    data = 255 - data;
    cv::minMaxLoc(data, nullptr, nullptr, nullptr, &max_loc);
    cv::imwrite("a.jpg", data);

    double center[2] = {max_loc.x * 1., max_loc.y * 1.};
    double sigma[2] = {1., 1.};
    double n[3] = {0., 0., 0.};
    double alpha = 0.;

    Problem problem;

    for (int i = 0; i < data.cols; ++i) {
        for (int j = 0; j < data.rows; ++j) {
            // if (pow(i - center[0], 2) + pow(j - center[1], 2) < 3*3) {
            CostFunction* cost_function =
                new AutoDiffCostFunction<ExponentialResidual, 1, 2, 2, 3, 1>(
                    new ExponentialResidual(i, j, data.at<uchar>(j, i) / 255.));
            problem.AddResidualBlock(cost_function, nullptr, center, sigma, n, &alpha);
            // std::cout << i << " " << j << std::endl;
            // }
        }
    }

    problem.SetParameterLowerBound(center, 1, 0.);
    problem.SetParameterUpperBound(center, 1, data.cols);

    Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    std::cout << " [" << center[0] << " " << center[1] << "] [" << sigma[0] << " " << sigma[1]
              << "] [" << n[0] << " " << n[1] << " " << n[2] << "] " << alpha << std::endl;

    // for (int i = 0; i < data.cols; ++i) {
    //     for (int j = 0; j < data.rows; ++j) {
    //         double out;
    //         auto res = ExponentialResidual(i, j, 0);
    //         res(center, sigma, n, &alpha, &out);
    //         data.at<uchar>(i, j) = std::abs(data.at<uchar>(i, j)-255 * out);
    //         std::cout << std::abs(data.at<uchar>(i, j)-255 * out) << std::endl;
    //     }
    // }
    center[0] += .5;
    center[1] += .5;
    cv::resize(data, data, data.size() * 8, cv::INTER_NEAREST_EXACT);
    data = 255 - data;
    cv::cvtColor(data, data, cv::COLOR_GRAY2BGR);

    cv::circle(data, cv::Point(center[0] * 8, center[1] * 8), 3, cv::Scalar(127, 127, 0));

    cv::imwrite("b.jpg", data);

    return 0;
}