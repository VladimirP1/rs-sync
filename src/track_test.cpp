
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "calibration.hpp"
#include "tracking.hpp"

class VideoReader {
   public:
    VideoReader(std::string filename) : cap_{filename} {}

    void SetPosition(double ms) {
        cap_.set(cv::CAP_PROP_POS_MSEC, ms);
        Advance();
    }

    bool Advance() {
        prev_gray_ = std::move(cur_gray_);
        prev_frame_ = std::move(cur_frame_);
        bool success = cap_.read(cur_frame_);
        cv::cvtColor(cur_frame_, cur_gray_, cv::COLOR_BGR2GRAY);
        return success;
    }

    const cv::Mat& Prev() const {
        return prev_frame_;
    }

    const cv::Mat& Cur() const {
        return cur_frame_;
    }

    const cv::Mat& PrevGray() const {
        return prev_gray_;
    }

    const cv::Mat& CurGray() const {
        return cur_gray_;
    }

   private:
   cv::Mat cur_gray_;
   cv::Mat prev_gray_;
    cv::Mat cur_frame_;
    cv::Mat prev_frame_;
    cv::VideoCapture cap_;
};

int main() {
    cv::VideoCapture cap("141101AA.MP4");
    // cv::VideoCapture cap("193653AA.MP4");

    cap.set(cv::CAP_PROP_POS_MSEC, 42e3);

    VideoReader reader("141101AA.MP4");
    TrackerImpl tracker(30, 700);
    FisheyeCalibration calibration;


    std::ifstream("GoPro_Hero6_2160p_43.json") >> calibration;

    reader.SetPosition(42e3);

    tracker.InitCorners(reader.CurGray());

    auto start = std::chrono::steady_clock::now();

    cv::Mat img;
    for (int i = 1; i < 300; ++i) {
        reader.Advance();
        auto [old_c, new_c] = tracker.Track(reader.PrevGray(), reader.CurGray());

        // img = reader.Cur().clone();

        // for (int i = 0; i < old_c.size(); ++i) {
        //     cv::line(img, old_c[i], new_c[i], cv::Scalar(0, 0, 255), 3);
        // }
        // cv::imwrite("outz" + std::to_string(i) + ".jpg", img);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dur = (end - start);
    std::cout
        << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
        << std::endl;

    return 0;
}