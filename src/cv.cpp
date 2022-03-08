#include "cv.hpp"

#include <unordered_map>
#include <fstream>
#include <cmath>

#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

void track_frames(FramesFlow& flow, Lens lens, const char* filename, int start_frame,
                  int end_frame) {
    cv::VideoCapture cap;
    if (!cap.open(filename)) {
        throw std::runtime_error{"video open failed"};
    }
    cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
    if (cap.get(cv::CAP_PROP_POS_FRAMES) != start_frame) {
        throw std::runtime_error{"Seek failed"};
    }
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create();

    cv::Mat next, cur, of;
    if (!cap.read(cur)) throw std::runtime_error{"frame read failed"};
    cv::cvtColor(cur, cur, cv::COLOR_BGR2GRAY);
    for (int frame = start_frame; frame < end_frame; ++frame) {
        std::cerr << "processing frame " << frame << std::endl;
        if (!cap.read(next)) throw std::runtime_error{"frame read failed"};
        cv::cvtColor(next, next, cv::COLOR_BGR2GRAY);

        dis->calc(cur, next, of);

        // Get OF at some points
        std::vector<arma::vec2> points_a, points_b;
        static constexpr int step = 200;
        for (int i = step; i < cur.cols; i += step) {
            for (int j = step; j < cur.rows; j += step) {
                points_a.push_back({i * 1., j * 1.});
                points_b.push_back({i + of.at<cv::Vec2f>(j, i)[0], j + of.at<cv::Vec2f>(j, i)[1]});
            }
        }

        // Undistort the points
        arma::mat& m = flow[frame];
        m.resize(8, points_a.size());
        for (int i = 0; i < points_a.size(); ++i) {
            arma::vec2 a = lens_undistort_point(lens, points_a[i]);
            arma::vec2 b = lens_undistort_point(lens, points_b[i]);

            double ts_a = (frame + 0.) / fps + lens.ro * (points_a[i][1] / cur.rows);
            double ts_b = (frame + 1.) / fps + lens.ro * (points_b[i][1] / cur.rows);

            m.submat(0, i, 1, i) = a;
            m.submat(3, i, 4, i) = b;
            m(2, i) = 1.;
            m(5, i) = 1.;
            m.submat(0, i, 2, i) = arma::normalise(m.submat(0, i, 2, i));
            m.submat(3, i, 5, i) = arma::normalise(m.submat(3, i, 5, i));
            m(6, i) = ts_a;
            m(7, i) = ts_b;

            // std::cout << m.col(i).t() << std::endl;
        }

        cur = std::move(next);
    }
}

/* Functions below do not depend on OpenCV */

Lens lens_load(const char* filename, const char* preset_name) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error{"Cannot open lens presets file"};

    Lens cl, ret{std::numeric_limits<double>::quiet_NaN()};
    while (in) {
        std::string cur_preset;
        in >> cur_preset;
        if (!in) break;
        in >> cl.ro >> cl.fx >> cl.fy >> cl.cx >> cl.cy >> cl.k1 >> cl.k2 >> cl.k3 >> cl.k4;
        // std::cout << cl.ro << " " << cl.fx << " " << cl.fy << " " << cl.cx << " " << cl.cy << " "
        // << cl.k1 << " " << cl.k2 << " " << cl.k3 << " " << cl.k4 << std::endl;
        if (cur_preset == preset_name) {
            ret = cl;
            break;
        }
    }
    if (std::isnan(ret.ro)) throw std::runtime_error{"Could not load preset"};
    return ret;
}

arma::vec2 lens_undistort_point(Lens lens, arma::vec2 point) {
    cv::Mat_<double> cam(3, 3), dist(4, 1);
    cam << lens.fx, 0., lens.cx, 0, lens.fy, lens.cy, 0, 0, 1;
    dist << lens.k1, lens.k2, lens.k3, lens.k4;

    std::vector<cv::Point2d> pts_in, pts_out;
    pts_in.push_back({point[0], point[1]});

    cv::fisheye::undistortPoints(pts_in, pts_out, cam, dist, cv::Mat::eye(3, 3, CV_32F));

    return arma::vec2({pts_out[0].x,pts_out[0].y});
}

// arma::vec2 lens_undistort_point(Lens lens, arma::vec2 point) {
//     static constexpr double eps = 1e-9;
//     static constexpr int kNumIterations = 9;

//     double x_ = (point[0] - lens.cx) / lens.fx;
//     double y_ = (point[1] - lens.cy) / lens.fy;
//     double theta_ = std::sqrt(x_ * x_ + y_ * y_);

//     double theta = M_PI / 4.;
//     for (int i = 0; i < kNumIterations; ++i) {
//         double theta2 = theta * theta, theta3 = theta2 * theta, theta4 = theta2 * theta2,
//                theta5 = theta2 * theta3, theta6 = theta3 * theta3, theta7 = theta3 * theta4,
//                theta8 = theta4 * theta4, theta9 = theta4 * theta5;
//         double cur_theta_ =
//             theta + lens.k1 * theta3 + lens.k2 * theta5 + lens.k3 * theta7 + lens.k4 * theta9;
//         double cur_dTheta_ = 1 + 3 * lens.k1 * theta2 + 5 * lens.k2 * theta4 +
//                              7 * lens.k3 * theta6 + 8 * lens.k4 * theta8;
//         double error = cur_theta_ - theta_;
//         double dthetaDtheta_ = 1. / cur_dTheta_;
//         double new_theta = theta - error * dthetaDtheta_;
//         while (new_theta >= M_PI / 2. || new_theta <= 0.) {
//             new_theta = (new_theta + theta) / 2.;
//         }
//         theta = new_theta;
//     }

//     double r = std::tan(theta);
//     double inv_cos_theta = 1. / std::cos(theta);
//     double s = (theta_ < eps) ? inv_cos_theta : r / theta_;

//     return {x_ * s, y_ * s};
// }

arma::vec2 lens_distort_point(Lens lens, arma::vec2 point) {
    double r = arma::norm(point);

    double theta = std::atan(r);

    double theta2 = theta * theta, theta3 = theta2 * theta, theta4 = theta2 * theta2,
           theta5 = theta2 * theta3, theta6 = theta3 * theta3, theta7 = theta3 * theta4,
           theta8 = theta4 * theta4, theta9 = theta4 * theta5;
    double theta_ =
        theta + lens.k1 * theta3 + lens.k2 * theta5 + lens.k3 * theta7 + lens.k4 * theta9;

    double k = theta_ / r;
    double x_ = point[0] * k, y_ = point[1] * k;
    double u = lens.fx * x_ + lens.cx, v = lens.fy * y_ + lens.cy;

    return {u, v};
}
