#pragma once

#include <armadillo>

inline void gyro_lowpass(arma::vec3* samples, int length, int divider);
inline void gyro_upsample(arma::vec3* samples, int length_new, int multiplier);
inline void gyro_decimate(arma::vec3* samples, int length, int divider);
int gyro_interpolate(arma::mat& timestamps, arma::mat& gyro);