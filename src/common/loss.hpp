#pragma once
#include "optdata.hpp"

void opt_compute_problem(int frame, double gyro_delay, const OptData& data, arma::mat& problem);
arma::vec3 opt_guess_translational_motion(const arma::mat& problem);