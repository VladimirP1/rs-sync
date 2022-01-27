#include "normal_fitter.hpp"

#include <ceres/ceres.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

namespace rssync {
void SubpixelMax(const cv::Mat& data_f, double& cx, double& cy);

struct Normal2dResidual {
    template <typename T>
    bool operator()(const T* ampl, const T* sigma, const T* alpha, T* residual) const {
        T xr = (x_ - cx_) * cos(*alpha) - (y_ - cy_) * sin(*alpha);
        T yr = (x_ - cx_) * sin(*alpha) + (y_ - cy_) * cos(*alpha);
        T xr2 = xr * xr, yr2 = yr * yr;

        residual[0] = ampl[0] * exp(-xr2 / sigma[0]) * exp(-yr2 / sigma[1]);
        residual[0] -= z_;

        // T d = xr2 / sigma[0] + yr2 / sigma[1];
        // residual[0] *= 1. - 1. / (1. + exp(-d));

        return true;
    }

    void SetXYZ(double cx, double cy, double x, double y, double z) {
        x_ = x;
        y_ = y;
        z_ = z;
        cx_ = cx;
        cy_ = cy;
    }

   private:
    double cx_{}, cy_{};
    double x_{}, y_{}, z_{};
};
class NormalFitter {
   public:
    NormalModel Fit(const cv::Mat& img) {
        FillProblem(img);
        Solver::Summary summary;
        Solve(options_, &problem_, &summary);
        // std::cout << sigma_[0] << " " << sigma_[1] << std::endl;
        return {A_, center_[0], center_[1], sigma_[0], sigma_[1], alpha_, 0.};
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
                    new AutoDiffCostFunction<Normal2dResidual, 1, 1, 2, 1>(residuals_.back());
                problem_.AddResidualBlock(cost_function, nullptr, &A_, sigma_, &alpha_);
            }
        }
        cur_width_ = width;
        cur_height_ = height;

        problem_.SetParameterLowerBound(sigma_, 0, 1e-3);
        problem_.SetParameterLowerBound(sigma_, 1, 1e-3);
        // problem_.SetParameterUpperBound(&alpha_, 0, 4. * M_PI);

        options_.max_num_iterations = 12;
        options_.linear_solver_type = ceres::DENSE_QR;
        options_.use_inner_iterations = false;
        options_.function_tolerance = 1e-4;
        options_.logging_type = ceres::SILENT;
        // options_.minimizer_progress_to_stdout = true;
    }

    void FillProblem(const cv::Mat& img) {
        double cx, cy;
        SubpixelMax(img, cx, cy);
        A_ = 1.;
        alpha_ = 0.;
        sigma_[0] = 1.;
        sigma_[1] = 2.;
        center_[0] = cx;
        center_[1] = cy;

        ConstructProblem(img.cols, img.rows);
        for (int j = 0; j < cur_height_; ++j) {
            for (int i = 0; i < cur_width_; ++i) {
                double z = img.at<float>(j, i);
                residuals_[i + j * cur_width_]->SetXYZ(cx, cy, i, j, z);
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

struct GaussianKernel {
    GaussianKernel(double x, double y, double sigma)
        : x_{x}, y_{y}, sigma_{sigma}, sigma2_m2_{sigma * sigma * 2} {}

    double Eval(double px, double py, double* dx = 0, double* dy = 0) {
        double exp_x = exp(-(px - x_) * (px - x_) / sigma2_m2_);
        double exp_y = exp(-(py - y_) * (py - y_) / sigma2_m2_);
        if (dx) {
            *dx += D * 2 * (px - x_) * exp_x / sigma2_m2_;
        }
        if (dy) {
            *dy += D * 2 * (py - y_) * exp_y / sigma2_m2_;
        }
        return (D * exp_x) * (D * exp_y);
    }

   private:
    double x_, y_, sigma_, sigma2_m2_;
    double D = 1. / sqrt(2 * M_PI) / sigma_;
};

class CubicBcKernel {
   public:
    CubicBcKernel(double B = 0., double C = .5)
        : P0{(6. - 2. * B) / 6.},
          P1{0.},
          P2{(-18. + 12. * B + 6. * C) / 6.},
          P3{(12. - 9. * B - 6. * C) / 6.},
          Q0{(8. * B + 24. * C) / 6.},
          Q1{(-12. * B - 48. * C) / 6.},
          Q2{(6. * B + 30. * C) / 6.},
          Q3{(-1. * B - 6. * C) / 6.} {}

    double Evaluate1D(double x) const {
        if (x < 0) x = -x;
        if (x < 1.) return P0 + P1 * x + P2 * x * x + P3 * x * x * x;
        if (x < 2.) return Q0 + Q1 * x + Q2 * x * x + Q3 * x * x * x;
        return 0.;
    }

    double Eval2D(double cx, double cy, double px, double py) const {
        double vx = Evaluate1D(px - cx);
        double vy = Evaluate1D(py - cy);
        return vx * vy;
    }

    double Eval(double cx, double cy, double px, double py, double* dx = 0, double* dy = 0) const {
        double dd = .01;
        if (dx) {
            *dx = (Eval2D(cx, cy, px - dd, py) - Eval2D(cx, cy, px + dd, py)) / dd / 2.;
        }
        if (dy) {
            *dy = (Eval2D(cx, cy, px, py - dd) - Eval2D(cx, cy, px, py + dd)) / dd / 2.;
        }
        return Eval2D(cx, cy, px, py);
    }

   private:
    double P0, P1, P2, P3, Q0, Q1, Q2, Q3;
};

void SubpixelMax(const cv::Mat& data_f, double& cx, double& cy) {
    cv::Point maxloc;
    cv::minMaxLoc(data_f, nullptr, nullptr, nullptr, &maxloc);
    int ix = maxloc.x;
    int iy = maxloc.y;

    static constexpr int ws = 3;
    std::vector<cv::Vec3d> img_pts;
    for (int j = std::max(0, iy - ws); j < std::min(data_f.rows, iy + ws + 1); ++j) {
        for (int i = std::max(0, ix - ws); i < std::min(data_f.cols, ix + ws + 1); ++i) {
            img_pts.push_back(
                {static_cast<double>(i), static_cast<double>(j), data_f.at<double>(j, i)});
            // std::cout << i << " " << j << " " << data_f.at<double>(j, i) << std::endl;
        }
    }

    CubicBcKernel k;
    static constexpr int iters = 10;
    static constexpr double alpha = .005;
    cx = ix;
    cy = iy;
    for (int i = 0; i < iters; ++i) {
        double sum{};
        double dx{}, dy{};
        for (const auto& p : img_pts) {
            double tdx, tdy;
            sum += k.Eval(cx, cy, p[0], p[1], &tdx, &tdy) * p[2];
            dx += tdx * p[2];
            dy += tdy * p[2];
        }
        double ss = log(1 + iters - i) / sqrt(dx * dx + dy * dy);
        // std::cout << sum << " " << dx << " " << dy << " " << cx << " " << cy << std::endl;
        // std::cout << log(1 + iters - i) * alpha << std::endl;
        cx += alpha * dx * ss;
        cy += alpha * dy * ss;
    }
}

class NormalFitterImpl : public INormalFitter {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {}

    NormalModel Fit(const cv::Mat& img) override { return fitter_.Fit(img); }

   private:
    NormalFitter fitter_;
};

void RegisterNormalFitter(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<NormalFitterImpl>(ctx, name);
}

}  // namespace rssync