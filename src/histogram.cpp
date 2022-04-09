#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap;
    // auto video = "GH011230.MP4";
    auto video = "PRO_VID_20210914_191851_00_006.mp4";
    if (!cap.open(video)) {
        throw std::runtime_error{"video open failed"};
    }
    int start_frame = 90;
    cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
    if (cap.get(cv::CAP_PROP_POS_FRAMES) != start_frame) {
        throw std::runtime_error{"Seek failed"};
    }
    double fps = cap.get(cv::CAP_PROP_FPS);

    while (1) {
        cv::Mat cur;
        if (!cap.read(cur)) {break;}
        cv::cvtColor(cur, cur, cv::COLOR_BGR2GRAY);

        int histSize = 64;
        float range[] = {0, 256};
        const float* histRange[] = {range};
        bool uniform = true, accumulate = false;
        cv::Mat hist;
        cv::calcHist(&cur, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate);

        for (int i = 0; i < hist.rows; ++i) {
            std::cout << hist.at<float>(i,0) << (i != hist.rows - 1 ? "," : "");
        }
        std::cout << std::endl;
    }
}