
#include <opencv2/core/hal/interface.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "tracking.hpp"


struct Tim {
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    ~Tim() {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> dur = (end - start);
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(dur)
                         .count()
                  << std::endl;
    }
};

void FastHarris(const cv::Mat& gray) {
    const int win_size = 5;
    const int half_win_size = win_size / 2;
    cv::Mat grad_x, grad_y;

    cv::Sobel(gray, grad_x, CV_16S, 1, 0);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1);

    cv::Mat grad_xx, grad_xy, grad_yy;

    const double k = 1. / 255.;
    cv::multiply(grad_x, grad_x, grad_xx, k);
    cv::multiply(grad_x, grad_y, grad_xy, k);
    cv::multiply(grad_y, grad_y, grad_yy, k);

    double min, max;
    cv::minMaxLoc(grad_xx, &min, &max);
    std::cout << min << " " << max << std::endl;

    {
    Tim t;
    for (int i = 0; i < grad_xx.rows; ++i) {
        int32_t prev_xx{0}, prev_xy{0}, prev_yy{0};
        for (int j = 1; j < grad_xx.cols; ++j) {
            prev_xx = (prev_xx + grad_xx.at<int16_t>(i, j));
            prev_xy = (prev_xy + grad_xy.at<int16_t>(i, j));
            prev_yy = (prev_yy + grad_yy.at<int16_t>(i, j));
            grad_xx.at<int16_t>(i, j) = prev_xx / win_size;
            grad_xy.at<int16_t>(i, j) = prev_xy / win_size;
            grad_yy.at<int16_t>(i, j) = prev_yy / win_size;
        }
    }
    }

    {
    Tim t;
    for (int j = 0; j < grad_xx.cols; ++j) {
        int32_t prev_xx{0}, prev_xy{0}, prev_yy{0};
        for (int i = 1; i < grad_xx.rows; ++i) {
            prev_xx = (prev_xx + grad_xx.at<int16_t>(i, j));
            prev_xy = (prev_xy + grad_xy.at<int16_t>(i, j));
            prev_yy = (prev_yy + grad_yy.at<int16_t>(i, j));
            grad_xx.at<int16_t>(i, j) = prev_xx / win_size;
            grad_xy.at<int16_t>(i, j) = prev_xy / win_size;
            grad_yy.at<int16_t>(i, j) = prev_yy / win_size;
        }
    }
    }

    // cv::Mat img;
    // cv::convertScaleAbs(grad_xx, img, 1 / 256. / 4);
    // cv::imwrite("out1.png", img);

    cv::Mat wgrad_xx = cv::Mat(grad_xx.rows, grad_xx.cols, CV_16S),
            wgrad_xy = cv::Mat(grad_xy.rows, grad_xy.cols, CV_16S),
            wgrad_yy = cv::Mat(grad_yy.rows, grad_yy.cols, CV_16S),
            response = cv::Mat::zeros(grad_yy.rows, grad_yy.cols, CV_16S);
    {
    Tim t;
    for (int i = win_size; i < grad_xx.rows; ++i) {
        for (int j = win_size; j < grad_xx.cols; ++j) {
            int m0 = wgrad_xx.at<int16_t>(i, j) =
                grad_xx.at<uint16_t>(i, j) -
                grad_xx.at<uint16_t>(i, j - win_size) -
                grad_xx.at<uint16_t>(i - win_size, j) +
                grad_xx.at<uint16_t>(i - win_size, j - win_size);
            int m1 = wgrad_xy.at<int16_t>(i, j) =
                grad_xy.at<uint16_t>(i, j) -
                grad_xy.at<uint16_t>(i, j - win_size) -
                grad_xy.at<uint16_t>(i - win_size, j) +
                grad_xy.at<uint16_t>(i - win_size, j - win_size);
            int m2 = wgrad_yy.at<int16_t>(i, j) =
                grad_yy.at<uint16_t>(i, j) -
                grad_yy.at<uint16_t>(i, j - win_size) -
                grad_yy.at<uint16_t>(i - win_size, j) +
                grad_yy.at<uint16_t>(i - win_size, j - win_size);

            response.at<int16_t>(i - half_win_size, j - half_win_size) =
                std::max(
                    (100 * (m0 * m2 - m1 * m1) - 5 * (m0 + m2) * (m0 + m2)) /
                        100 / 256,
                    0);
        }
    }
    }

    cv::minMaxLoc(response, &min, &max);
    std::cout << min << " " << max << std::endl;

    cv::Mat img;
    cv::convertScaleAbs(response, img, 1);
    cv::imwrite("out2.png", img);

    // auto m0 = cv::sum(grad_x.mul(grad_x))[0];
    // auto m1 = cv::sum(grad_x.mul(grad_y))[0];
    // auto m2 = cv::sum(grad_y.mul(grad_y))[0];

    // const auto k = .05;
    // auto response = m0 * m2 - m1 * m1 - k * (m0 + m2) * (m0 + m2);
    // return std::max(response, 0.);
}

int main() {
    cv::VideoCapture cap("141101AA.MP4");

    cap.set(cv::CAP_PROP_POS_MSEC, 42e3);

    cv::Mat img;
    cv::Mat gray;

    TrackerImpl t(30, 300);

    cap.read(img);
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    auto start = std::chrono::steady_clock::now();

    FastHarris(gray);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dur = (end - start);
    std::cout
        << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
        << std::endl;

    return 0;
}

