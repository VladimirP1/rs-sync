#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>
#include <iostream>

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

int main(int argc, char** argv) {
    cv::Mat data = cv::imread(argv[1]);
    cv::cvtColor(data, data, cv::COLOR_BGR2GRAY);
    cv::Mat data_f;
    data.convertTo(data_f, CV_64F);
    data_f /= cv::sum(data_f)[0];

    double cx, cy;

    for(int i = 0; i < 100000; ++i) {
    SubpixelMax(data_f, cx, cy);
    }

    std::cout << cx << " " << cy << std::endl;

    cv::resize(data, data, data.size() * 10, cv::INTER_CUBIC);
    cv::circle(data, cv::Point((cx + .5) * 10, (cy + .5) * 10), 1, 0, -1, cv::LINE_AA);
    cv::cvtColor(data, data, cv::COLOR_GRAY2BGR);
    cv::imwrite("b.jpg", data);

    return 0;
}