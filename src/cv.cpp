
#include <opencv2/core/hal/interface.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/intrin_sse.hpp>
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
    constexpr int win_size_pow2 = 4;
    constexpr int win_size = 1 << win_size_pow2;
    constexpr int half_win_size = win_size / 2;
    cv::Mat grad_x, grad_y;

    {
        Tim t;
    cv::Sobel(gray, grad_x, CV_16S, 1, 0);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1);
    }

    cv::Mat grad_xx, grad_xy, grad_yy;

    const double k = 1. / 255.;
    cv::multiply(grad_x, grad_x, grad_xx, k);
    cv::multiply(grad_x, grad_y, grad_xy, k);
    cv::multiply(grad_y, grad_y, grad_yy, k);

    // double min, max;
    // cv::minMaxLoc(grad_xx, &min, &max);
    // std::cout << min << " " << max << std::endl;

    {
        Tim t;
        for (int i = 0; i < grad_xx.rows; ++i) {
            int32_t prev_xx{0}, prev_xy{0}, prev_yy{0};
            int16_t* row_xx = reinterpret_cast<int16_t*>(grad_xx.ptr(i));
            int16_t* row_xy = reinterpret_cast<int16_t*>(grad_xy.ptr(i));
            int16_t* row_yy = reinterpret_cast<int16_t*>(grad_yy.ptr(i));
            for (int j = 1; j < grad_xx.cols; ++j) {
                prev_xx = (prev_xx + row_xx[j]);
                prev_xy = (prev_xy + row_xy[j]);
                prev_yy = (prev_yy + row_yy[j]);
                row_xx[j] = prev_xx >> 2 * win_size_pow2;
                row_xy[j] = prev_xy >> 2 * win_size_pow2;
                row_yy[j] = prev_yy >> 2 * win_size_pow2;
            }
        }
    }

    {
        Tim t;
        for (int j = 0; j < grad_xx.cols; j += 8) {
            cv::v_int16x8 prev_xx = cv::v_setzero_s16(), prev_xy = cv::v_setzero_s16(), prev_yy = cv::v_setzero_s16();
            for (int i = 1; i < grad_xx.rows; ++i) {
                int16_t* row_xx = reinterpret_cast<int16_t*>(grad_xx.ptr(i));
                int16_t* row_xy = reinterpret_cast<int16_t*>(grad_xy.ptr(i));
                int16_t* row_yy = reinterpret_cast<int16_t*>(grad_yy.ptr(i));
                prev_xx = cv::v_add_wrap(prev_xx, cv::v_load(row_xx + j));
                prev_xy = cv::v_add_wrap(prev_xy, cv::v_load(row_xy + j));
                prev_yy = cv::v_add_wrap(prev_yy, cv::v_load(row_yy + j));
                cv::v_store(row_xx + j, prev_xx);
                cv::v_store(row_xy + j, prev_xy);
                cv::v_store(row_yy + j, prev_yy);
            }
        }
    }

    // cv::Mat img;
    // cv::convertScaleAbs(grad_xx, img, 1 / 256. / 4);
    // cv::imwrite("out1.png", img);

    cv::Mat response = cv::Mat::zeros(grad_yy.rows, grad_yy.cols, CV_16S);
    {
        Tim t;
        for (int i = win_size; i < grad_xx.rows; ++i) {
            uint16_t* row_xx = reinterpret_cast<uint16_t*>(grad_xx.ptr(i));
            uint16_t* row_xy = reinterpret_cast<uint16_t*>(grad_xy.ptr(i));
            uint16_t* row_yy = reinterpret_cast<uint16_t*>(grad_yy.ptr(i));
            uint16_t* mrow_xx = reinterpret_cast<uint16_t*>(grad_xx.ptr(i - win_size));
            uint16_t* mrow_xy = reinterpret_cast<uint16_t*>(grad_xy.ptr(i - win_size));
            uint16_t* mrow_yy = reinterpret_cast<uint16_t*>(grad_yy.ptr(i - win_size));
            for (int j = win_size; j < grad_xx.cols; ++j) {
                int m0 = static_cast<int16_t>(row_xx[j] - row_xx[j - win_size] - mrow_xx[j] + mrow_xx[j - win_size]);
                int m1 = static_cast<int16_t>(row_xy[j] - row_xy[j - win_size] - mrow_xy[j] + mrow_xy[j - win_size]);
                int m2 = static_cast<int16_t>(row_yy[j] - row_yy[j - win_size] - mrow_yy[j] + mrow_yy[j - win_size]);

                response.at<int16_t>(i - half_win_size, j - half_win_size) = 
                    std::max((100 * (m0 * m2 - m1 * m1) -
                              5 * (m0 + m2) * (m0 + m2)) /
                                 100 / 256,
                             0);
            }
        }
    }

    // cv::minMaxLoc(response, &min, &max);
    // std::cout << min << " " << max << std::endl;
}

int main() {
    cv::VideoCapture cap("141101AA.MP4");

    cap.set(cv::CAP_PROP_POS_MSEC, 42e3);

    cv::Mat img;
    cv::Mat gray;

    TrackerImpl t(30, 300);

    cap.read(img);
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    {
    Tim t;
    for (int i = 0; i < 30; ++i) {
        FastHarris(gray);
    }
    }

    {
    Tim t;
    for (int i = 0; i < 30; ++i) {
        std::vector<cv::Point> pts;
        cv::goodFeaturesToTrack(gray, pts, 500, .01, 0);
    }
    }


    return 0;
}
