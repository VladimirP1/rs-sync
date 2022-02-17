#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

using namespace cv;
using namespace std;
static Mat getVisibleFlow(InputArray flow) {
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.5);
    vector<UMat> hsv_vec;
    hsv_vec.push_back(angle);
    hsv_vec.push_back(UMat::ones(angle.size(), angle.type()));
    hsv_vec.push_back(magnitude);
    UMat hsv;
    merge(hsv_vec, hsv);
    Mat img;
    cvtColor(hsv, img, COLOR_HSV2BGR);
    return img;
}

int main(int argc, char** argv) {
    cv::VideoCapture cap("000458AA.MP4");

    cap.set(cv::CAP_PROP_POS_FRAMES, 1140);

    for (int i = 0; i < 100; ++i) {
        std::cout << i << std::endl;
        cv::Mat img_a, img_b, flow;
        cap.read(img_a);
        cap.read(img_b);

        cvtColor(img_a, img_a, cv::COLOR_BGR2GRAY);
        cvtColor(img_b, img_b, cv::COLOR_BGR2GRAY);

        auto of = cv::DISOpticalFlow::create();
        of->calc(img_a, img_b, flow);

        // cv::imwrite("out_" + std::to_string(i) + ".png", getVisibleFlow(flow));
    }
}