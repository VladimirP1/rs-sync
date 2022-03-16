#pragma once

#include <armadillo>
#include <unordered_map>

struct Lens {
    double ro{};
    double fx{}, fy{};
    double cx{}, cy{};
    double k1{}, k2{}, k3{}, k4{};
};

using FramesFlow = std::unordered_map<int, arma::mat>;

Lens lens_load(const char* filename, const char* preset_name);
void track_frames(FramesFlow& flow, Lens lens, const char* filename, int start_frame, int end_frame);
arma::vec2 lens_undistort_point(Lens lens, arma::vec2 point);
arma::vec2 lens_distort_point(Lens lens, arma::vec2 point);