#pragma once
#include "optdata.hpp"

void opt_compute_problem(int frame, double gyro_delay, const OptData& data, arma::mat& problem);
arma::vec3 opt_guess_translational_motion(const arma::mat& problem, int max_iters = 200);
std::pair<double, double> pre_sync(OptData& opt_data, int frame_begin, int frame_end, double rough_delay,
                double search_radius, double step);