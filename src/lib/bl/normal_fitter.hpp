#pragma once
#include "component.hpp"

#include <cmath>

#include <opencv2/core.hpp>

namespace rssync {

class NormalModel {
   public:
    NormalModel() {}
    NormalModel(double a, double cx, double cy, double sx, double sy, double alpha, double offset)
        : a_{a}, cx_{cx}, cy_{cy}, sx_{sx}, sy_{sy}, alpha_{alpha}, offset_{offset} {}

    double Evaluate(double x, double y) const {
        double xr = (x - cx_) * cos(alpha_) - (y - cy_) * sin(alpha_);
        double yr = (x - cx_) * sin(alpha_) + (y - cy_) * cos(alpha_);
        double xr2 = xr * xr, yr2 = yr * yr;

        return a_ * exp(-xr2 / sx_) * exp(-yr2 / sy_) + offset_;
    }

    void GetCenter(double& cx, double& cy) const {
        cx = cx_;
        cy = cy_;
    }

    void ShiftOrigin(double x, double y) {
        cx_ += x;
        cy_ += y;
    }

   private:
    double cx_, cy_;
    double sx_, sy_;
    double a_, alpha_, offset_{};
};

class INormalFitter : public rssync::BaseComponent {
   public:
    virtual NormalModel Fit(const cv::Mat& img) = 0;
};

void RegisterNormalFitter(std::shared_ptr<IContext> ctx, std::string name);

constexpr const char* kNormalFitterName = "NormalFitter";
}  // namespace rssync