#pragma once
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

class VideoReader {
   public:
    VideoReader(const char* filename) : cap_{filename} {}

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

    const cv::Mat& Prev() const { return prev_frame_; }

    const cv::Mat& Cur() const { return cur_frame_; }

    const cv::Mat& PrevGray() const { return prev_gray_; }

    const cv::Mat& CurGray() const { return cur_gray_; }

   private:
    cv::Mat cur_gray_;
    cv::Mat prev_gray_;
    cv::Mat cur_frame_;
    cv::Mat prev_frame_;
    cv::VideoCapture cap_;
};