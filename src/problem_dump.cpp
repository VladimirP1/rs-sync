#include <execution>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>
#include <filesystem>

#include "cv.hpp"
#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "backtrack.hpp"
#include "simple_calculus.hpp"
#include "gsl.hpp"
#include "loss.hpp"
#include "io.hpp"



int main() {
    std::cout << std::fixed << std::setprecision(14);

    auto frame_start = 600, frame_end = 700;
    double gyro_delay = -42;
    // double gyro_delay = 1000;

    OptData opt_data;
    optdata_fill_gyro(opt_data, "GX011338.MP4", "yZX");
    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    track_frames(opt_data.flows, lens, "GX011338.MP4", frame_start, frame_end);

    std::filesystem::create_directories("dump");

    std::ofstream motion_stream("dump/motion.csv");

    arma::mat problem;
    for (int frame = frame_start; frame < frame_end; ++frame) {
        opt_compute_problem(frame, gyro_delay, opt_data, problem);

        std::ofstream normals_stream("dump/normals_" + std::to_string(frame) + ".csv");
        for (int i = 0; i < problem.n_rows; ++i) {
            auto row = problem.row(i).eval();
            normals_stream << row(0) << "," << row(1) << "," << row(2) << "\n";
        }

        auto motion = opt_guess_translational_motion(problem);
        motion_stream << frame << "," << motion(0) << "," << motion(1) << "," << motion(2) << "\n";
    }
}
