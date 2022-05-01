#include <rssync.h>

#include <quat.hpp>
#include <signal.hpp>

#include <telemetry-parser.h>

#include <armadillo>

#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <nlohmann/json.hpp>

// Either fixed-sample-rate or variable sr data can be passed to the library (in the latter case it
// will be interpolated internally)
#if 0
void optdata_fill_gyro(ISyncProblem& problem, const char* filename, const char* orient) {
    tp_gyrodata data = tp_load_gyro(filename, orient);
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    tp_free(data);
    double sample_rate = gyro_interpolate(timestamps, gyro);

    arma::mat quats(4, gyro.n_cols);
    quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < quats.n_cols; ++i) {
        quats.col(i) =
            arma::normalise(quat_prod(quat_from_aa(gyro.col(i) / sample_rate), quats.col(i - 1)));
    }
    problem.SetGyroQuaternions(quats.mem, quats.n_cols, sample_rate, timestamps.front());
}
#else
void optdata_fill_gyro(ISyncProblem& problem, const char* filename, const char* orient) {
    tp_gyrodata data = tp_load_gyro(filename, orient);
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    arma::mat quats(4, gyro.n_cols);
    quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < quats.n_cols; ++i) {
        auto q = quat_from_aa(gyro.col(i) * (timestamps(i) - timestamps(i - 1)));
        quats.col(i) = arma::normalise(quat_prod(q, quats.col(i - 1)));
    }

    std::vector<uint64_t> i_timestamps;
    for (auto ts : timestamps) {
        i_timestamps.push_back(ts * 1000000);
    }
    problem.SetGyroQuaternions(i_timestamps.data(), quats.mem, i_timestamps.size());
}
#endif

struct Lens {
    double ro{};
    double fx{}, fy{};
    double cx{}, cy{};
    double k1{}, k2{}, k3{}, k4{};
};

arma::vec2 lens_undistort_point(Lens lens, arma::vec2 point) {
    if (arma::norm(point) < 1e-8) return {0, 0};
    static constexpr double eps = 1e-9;
    static constexpr int kNumIterations = 9;

    double x_ = (point[0] - lens.cx) / lens.fx;
    double y_ = (point[1] - lens.cy) / lens.fy;
    double theta_ = std::sqrt(x_ * x_ + y_ * y_);

    double theta = M_PI / 4.;
    for (int i = 0; i < kNumIterations; ++i) {
        double theta2 = theta * theta, theta3 = theta2 * theta, theta4 = theta2 * theta2,
               theta5 = theta2 * theta3, theta6 = theta3 * theta3, theta7 = theta3 * theta4,
               theta8 = theta4 * theta4, theta9 = theta4 * theta5;
        double cur_theta_ =
            theta + lens.k1 * theta3 + lens.k2 * theta5 + lens.k3 * theta7 + lens.k4 * theta9;
        double cur_dTheta_ = 1 + 3 * lens.k1 * theta2 + 5 * lens.k2 * theta4 +
                             7 * lens.k3 * theta6 + 8 * lens.k4 * theta8;
        double error = cur_theta_ - theta_;
        double dthetaDtheta_ = 1. / cur_dTheta_;
        double new_theta = theta - error * dthetaDtheta_;
        while (new_theta >= M_PI / 2. || new_theta <= 0.) {
            new_theta = (new_theta + theta) / 2.;
        }
        theta = new_theta;
    }

    double r = std::tan(theta);
    double inv_cos_theta = 1. / std::cos(theta);
    double s = (theta_ < eps) ? inv_cos_theta : r / theta_;

    return {x_ * s, y_ * s};
}

void track_frames(ISyncProblem& problem, Lens lens, const char* filename, int start_frame,
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
        arma::mat points3d_a(3, points_a.size());
        arma::mat points3d_b(3, points_b.size());
        arma::mat tss_a(1, points_a.size());
        arma::mat tss_b(1, points_b.size());

        for (int i = 0; i < points_a.size(); ++i) {
            arma::vec2 a = lens_undistort_point(lens, points_a[i]);
            arma::vec2 b = lens_undistort_point(lens, points_b[i]);

            double ts_a = (frame + 0.) / fps + lens.ro * (points_a[i][1] / cur.rows);
            double ts_b = (frame + 1.) / fps + lens.ro * (points_b[i][1] / cur.rows);

            points3d_a.submat(0, i, 1, i) = a;
            points3d_b.submat(0, i, 1, i) = b;
            points3d_a(2, i) = 1.;
            points3d_b(2, i) = 1.;
            points3d_a.col(i) = arma::normalise(points3d_a.col(i));
            points3d_b.col(i) = arma::normalise(points3d_b.col(i));
            tss_a(0, i) = ts_a;
            tss_b(0, i) = ts_b;
        }

        problem.SetTrackResult(frame, tss_a.mem, tss_b.mem, points3d_a.mem, points3d_b.mem,
                               points_a.size());
        cur = std::move(next);
    }
}

Lens lens_load(const char* filename, const char* preset_name) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error{"Cannot open lens presets file"};

    Lens cl, ret{std::numeric_limits<double>::quiet_NaN()};
    while (in) {
        std::string cur_preset;
        in >> cur_preset;
        if (!in) break;
        in >> cl.ro >> cl.fx >> cl.fy >> cl.cx >> cl.cy >> cl.k1 >> cl.k2 >> cl.k3 >> cl.k4;
        if (cur_preset == preset_name) {
            ret = cl;
            break;
        }
    }
    if (std::isnan(ret.ro)) throw std::runtime_error{"Could not load preset"};
    return ret;
}

using json = nlohmann::json;

int main(int argc, char** argv) {
    // ISyncProblem* sp = CreateSyncProblem();

    // optdata_fill_gyro(*sp, "GH011230.MP4", "yZX");
    // Lens lens = lens_load("lens.txt", "hero6_27k_43");
    // track_frames(*sp, lens, "GH011230.MP4", 90, 200);

    // double sync = sp->PreSync(0, 90, 150, 1e-2, .2);
    // for (int i = 0; i < 4; ++i) sync = sp->Sync(sync, 90, 150);

    // std::cout << sync << std::endl;

    //------------
    std::unique_ptr<ISyncProblem> sp{CreateSyncProblem()};

    std::ifstream ifs(argv[1]);
    json j = json::parse(ifs);

    json input = j["input"];
    json params = j["params"];
    json output = j["output"];

    optdata_fill_gyro(*sp, input["gyro_path"].get<std::string>().c_str(),
                      input["gyro_orientation"].get<std::string>().c_str());

    Lens lens = lens_load(input["lens_profile"]["path"].get<std::string>().c_str(),
                          input["lens_profile"]["name"].get<std::string>().c_str());

    int frame_start = input["frame_range"][0].get<int>();
    int frame_end = input["frame_range"][1].get<int>();
    int sync_window = params["sync_window"].get<int>();
    int syncpoint_distance = params["syncpoint_distance"].get<int>();
    track_frames(*sp, lens, input["video_path"].get<std::string>().c_str(),
                 input["frame_range"][0].get<int>(), input["frame_range"][1].get<int>());

    std::vector<int> syncpoints;
    if (params["syncpoints_format"] == "auto") {
        for (int pos = frame_start; pos + sync_window < frame_end; pos += syncpoint_distance)
            syncpoints.push_back(pos);
    } else if (params["syncpoints_format"] == "array") {
        for (int pos : params["syncpoints_array"]) {
            syncpoints.push_back(pos);
        }
    } else {
        return 1;
    }

    std::ofstream csv(output["csv_path"].get<std::string>());

// Demonstration of how to export the cost plot from pre-sync
#if 1
    {
        const size_t debug_plot_size = 200;
        std::vector<double> delays(debug_plot_size);
        std::vector<double> costs(debug_plot_size);

        sp->DebugPreSync(input["initial_guess"].get<double>() / 1000, frame_start,
                         frame_start + sync_window,
                         input["simple_presync_radius"].get<double>() / 1000, delays.data(),
                         costs.data(), debug_plot_size);

        std::ofstream debug_out("debug.csv");
        for (size_t i = 0; i < debug_plot_size; ++i) {
            debug_out << delays[i] << "," << costs[i] << "\n";
        }
    }
#endif

    for (auto pos : syncpoints) {
        std::cerr << pos << std::endl;
        double delay = input["initial_guess"].get<double>() / 1000;
        if (input.contains("use_simple_presync") && input["use_simple_presync"].get<bool>()) {
            delay = sp->PreSync(delay, pos, pos + sync_window,
                                input["simple_presync_step"].get<double>() / 1000.,
                                input["simple_presync_radius"].get<double>() / 1000.);
        }
        for (int i = 0; i < 4; ++i) delay = sp->Sync(delay, pos, pos + sync_window);
        csv << pos << "," << 1000 * delay << std::endl;
    }

    csv.close();
}