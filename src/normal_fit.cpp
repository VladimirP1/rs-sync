#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

void residual(double x, double y, double z, double* params, double* value, double* jacobians) {
    double center_x = params[0], center_y = params[1], sigma_x = params[2], sigma_y = params[3],
           ampl = params[4], alpha = params[5];
    // clang-format off
    double xr = (x - center_x) * cos(alpha) - (y - center_y) * sin(alpha);
    double yr = (x - center_x) * sin(alpha) + (y - center_y) * cos(alpha);
    double dxrDcenter_x = -(x - center_x) * cos(alpha);
    double dyrDcenter_x = -(x - center_x) * sin(alpha);
    double dxrDcenter_y = (y - center_y) * sin(alpha);
    double dyrDcenter_y = -(y - center_y) * cos(alpha);
    double dxrDalpha = -(x - center_x) * sin(alpha) - (y - center_y) * cos(alpha);
    double dyrDalpha = (x - center_x) * cos(alpha) - (y - center_y) * sin(alpha);

    double xr2 = xr * xr, yr2 = yr * yr;
    double dxr2Dxr = 2. * xr;
    double dyr2Dyr = 2. * yr;

    double d = xr2 + yr2;
    double ddDcenter_x = -2. * (x - center_x);
    double ddDcenter_y = -2. * (y - center_y);

    double scale = 1. - 1. / (1. + exp(-d));
    double dscaleDd = -exp(-d) * (scale - 1) * (scale - 1);

    double val = (ampl * exp(-xr2 / sigma_x) * exp(-yr2 / sigma_y) - z) * scale;
    double dvalueDampl = scale * exp(-xr2 / sigma_x) * exp(-yr2 / sigma_y);
    double dvalueDxr2 = -ampl * scale * exp(-xr2 / sigma_x) * exp(-yr2 / sigma_y) / sigma_x;
    double dvalueDyr2 = -ampl * scale * exp(-xr2 / sigma_x) * exp(-yr2 / sigma_y) / sigma_y;
    double dvalueDsigma_x = ampl * scale * xr2 * exp(-xr2 / sigma_x) * exp(-yr2 / sigma_y) / sigma_x / sigma_x;
    double dvalueDsigma_y = ampl * scale * yr2 * exp(-xr2 / sigma_x) * exp(-yr2 / sigma_y) / sigma_y / sigma_y;
    double dvalueDscale = ampl * exp(-xr2 / sigma_x) * exp(-yr2 / sigma_y) - z;

    /* center_x */   jacobians[0] += dvalueDxr2 * dxr2Dxr * dxrDcenter_x + dvalueDyr2 * dyr2Dyr * dyrDcenter_x + dvalueDscale * dscaleDd * ddDcenter_x;
    /* center_y */   jacobians[1] += dvalueDxr2 * dxr2Dxr * dxrDcenter_y + dvalueDyr2 * dyr2Dyr * dyrDcenter_y + dvalueDscale * dscaleDd * ddDcenter_y;
    /* sigma_x  */   jacobians[2] += dvalueDsigma_x;
    /* sigma_y  */   jacobians[3] += dvalueDsigma_y;
    /* ampl     */   jacobians[4] += dvalueDampl;
    /* alpha    */   jacobians[5] += dvalueDxr2 * dxr2Dxr * dxrDalpha + dvalueDyr2 * dyr2Dyr * dyrDalpha;

    value[0] += val * val;
    // clang-format on
}

int main(int argc, char** argv) {
    cv::Point max_loc;
    cv::Mat data = cv::imread(argv[1]);
    cv::cvtColor(data, data, cv::COLOR_BGR2GRAY);
    data = 255 - data;
    cv::minMaxLoc(data, nullptr, nullptr, nullptr, &max_loc);
    cv::imwrite("a.jpg", data);

    double res;
    double jacobians[6], tmp[6];
    double params[] = {max_loc.x * 1., max_loc.y * 1., 1., 1., 1., 0.};

    double rate = 1e-1;

    for (int k = 0; k < 100; ++k) {
        res = 0;
        std::fill(jacobians, jacobians + 6, 0.);
        for (int i = 0; i < data.cols; ++i) {
            for (int j = 0; j < data.rows; ++j) {
                double z = data.at<uchar>(j, i) / 255.;
                residual(i, j, z, params, &res, jacobians);
            }
        }
        for (int i = 0; i < 6; ++i) {
            if (i != 2 && i!= 3 && i!=4)
            params[i] -= jacobians[i] * rate;
        }
        std::cout << res << " " << params[0] << " " << params[1] << std::endl;
    }

    return 0;
}