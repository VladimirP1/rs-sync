#pragma once

#include <armadillo>

void gyro_lowpass(arma::mat& samples, int divider);
void gyro_upsample(arma::mat& samples, int multiplier);
void gyro_decimate(arma::mat& samples, int divider);
int gyro_interpolate(arma::mat& timestamps, arma::mat& gyro);