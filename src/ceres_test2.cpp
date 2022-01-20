#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>

struct Normal2dResidual {
    template <typename T>
    bool operator()(const T* ampl, const T* const center, const T* sigma, const T* alpha,
                    T* residual) const {
        T xr = (x_ - center[0]) * cos(*alpha) - (y_ - center[1]) * sin(*alpha);
        T yr = (x_ - center[0]) * sin(*alpha) + (y_ - center[1]) * cos(*alpha);
        T xr2 = xr * xr, yr2 = yr * yr;

        residual[0] = ampl[0] * exp(-xr2 / sigma[0]) * exp(-yr2 / sigma[1]);
        residual[0] -= z_;

        T d = xr2 / sigma[0] + yr2 / sigma[1];
        residual[0] *= 1. - 1. / (1. + exp(-d));

        return true;
    }

    void SetXYZ(double x, double y, double z) {
        x_ = x;
        y_ = y;
        z_ = z;
    }

   private:
    double x_{};
    double y_{};
    double z_{};
};

class NormalModel {
   public:
    NormalModel(double a, double cx, double cy, double sx, double sy, double alpha)
        : a_{a}, cx_{cx}, cy_{cy}, sx_{sx}, sy_{sy}, alpha_{alpha} {}

    double Evaluate(double x, double y) const {
        double xr = (x - cx_) * cos(alpha_) - (y - cy_) * sin(alpha_);
        double yr = (x - cy_) * sin(alpha_) + (y - cy_) * cos(alpha_);
        double xr2 = xr * xr, yr2 = yr * yr;
        double d = xr2 + yr2;

        return a_ * exp(-xr2 / sx_) * exp(-yr2 / sy_);
    }

    void GetCenter(double& cx, double& cy) const {
        cx = cx_;
        cy = cy_;
    }

   private:
    const double cx_, cy_;
    const double sx_, sy_;
    const double a_, alpha_;
};

class NormalFitter {
   public:
    NormalModel Fit(const cv::Mat& img) {
        FillProblem(img);
        Solver::Summary summary;
        Solve(options_, &problem_, &summary);
        return {A_, center_[0], center_[1], sigma_[0], sigma_[1], alpha_};
    }

   private:
    void ConstructProblem(int width, int height) {
        if (width == cur_width_ && height == cur_height_) {
            return;
        }
        problem_ = {};
        residuals_.clear();
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                residuals_.push_back(new Normal2dResidual());
                CostFunction* cost_function =
                    new AutoDiffCostFunction<Normal2dResidual, 1, 1, 2, 2, 1>(residuals_.back());
                problem_.AddResidualBlock(cost_function, nullptr, &A_, center_, sigma_, &alpha_);
            }
        }
        cur_width_ = width;
        cur_height_ = height;

        problem_.SetParameterLowerBound(center_, 0, 0.);
        problem_.SetParameterUpperBound(center_, 0, cur_width_);
        problem_.SetParameterLowerBound(center_, 1, 0.);
        problem_.SetParameterUpperBound(center_, 1, cur_height_);
        problem_.SetParameterLowerBound(&alpha_, 0, 0.);
        problem_.SetParameterUpperBound(&alpha_, 0, 2. * M_PI);

        options_.max_num_iterations = 8;
        options_.linear_solver_type = ceres::DENSE_QR;
        options_.use_inner_iterations = false;
        options_.logging_type = ceres::SILENT;
        // options_.minimizer_progress_to_stdout = true;
    }

    void FillProblem(const cv::Mat& img) {
        cv::Point max_loc;
        cv::minMaxLoc(img, nullptr, nullptr, nullptr, &max_loc);
        A_ = 1.;
        alpha_ = 0.;
        center_[0] = max_loc.x * 1.;
        center_[1] = max_loc.y * 1.;
        sigma_[0] = sigma_[1] = 1.;

        ConstructProblem(img.cols, img.rows);
        for (int j = 0; j < cur_height_; ++j) {
            for (int i = 0; i < cur_width_; ++i) {
                double z = img.at<uchar>(j, i) / 255.;
                residuals_[i + j * cur_width_]->SetXYZ(i, j, z);
            }
        }
    }

   private:
    Problem problem_;
    Solver::Options options_;
    double A_, alpha_;
    double center_[2];
    double sigma_[2];
    std::vector<Normal2dResidual*> residuals_;
    int cur_width_{-1}, cur_height_{-1};
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    cv::Point max_loc;
    cv::Mat data = cv::imread(argv[1]);
    cv::cvtColor(data, data, cv::COLOR_BGR2GRAY);
    data = 255 - data;
    cv::minMaxLoc(data, nullptr, nullptr, nullptr, &max_loc);
    cv::imwrite("a.jpg", data);

    NormalFitter fitter;

    for (int i = 0; i < 1000; ++i) {
        fitter.Fit(data);
    }
    // std::cout << summary.BriefReport() << "\n";

    cv::resize(data, data, data.size() * 8, cv::INTER_CUBIC);
    data = 255 - data;
    cv::cvtColor(data, data, cv::COLOR_GRAY2BGR);

    // cv::circle(data, cv::Point(center[0] * 8, center[1] * 8), 3, cv::Scalar(127, 127, 0));

    cv::imwrite("b.jpg", data);

    return 0;
}