#include "normal_fitter.hpp"

#include <ceres/ceres.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

namespace rssync {
struct Normal2dResidual {
    template <typename T>
    bool operator()(const T* ampl, const T* const center, const T* sigma, const T* alpha,
                    T* residual) const {
        T xr = (x_ - center[0]) * cos(*alpha) - (y_ - center[1]) * sin(*alpha);
        T yr = (x_ - center[0]) * sin(*alpha) + (y_ - center[1]) * cos(*alpha);
        T xr2 = xr * xr, yr2 = yr * yr;

        residual[0] = ampl[0] * exp(-xr2 / sigma[0]) * exp(-yr2 / sigma[1]);
        residual[0] -= z_;

        T d = xr2 + yr2;
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

class NormalFitterImpl : public INormalFitter {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {}

    NormalModel Fit(const cv::Mat& img, double offset) override {
        FillProblem(img);
        Solver::Summary summary;
        Solve(options_, &problem_, &summary);
        return {A_, center_[0], center_[1], sigma_[0], sigma_[1], alpha_, offset};
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
        problem_.SetParameterLowerBound(sigma_, 0, 1e-3);
        problem_.SetParameterLowerBound(sigma_, 1, 1e-3);
        // problem_.SetParameterLowerBound(&alpha_, 0, 0.);
        // problem_.SetParameterUpperBound(&alpha_, 0, 2. * M_PI);

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
                double z = img.at<float>(j, i);
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

void RegisterNormalFitter(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<NormalFitterImpl>(ctx, name);
}

}  // namespace rssync