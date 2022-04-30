#include "signal.hpp"

void gyro_lowpass(arma::mat& samples, int divider) {
    if (divider < 2) return;
    const double ita = 1.0 / tan(M_PI / divider);
    const double q = sqrt(2.0);
    const double b0 = 1.0 / (1.0 + q * ita + ita * ita), b1 = 2 * b0, b2 = b0,
                 a1 = 2.0 * (ita * ita - 1.0) * b0, a2 = -(1.0 - q * ita + ita * ita) * b0;

    arma::vec3 out[3] = {samples.col(0), samples.col(1), samples.col(2)};
    for (int i = 2; i < samples.n_cols; ++i) {
        out[2] = b0 * samples.col(i) + b1 * samples.col(i - 1) + b2 * samples.col(i - 2) +
                 a1 * out[2 - 1] + a2 * out[2 - 2];
        samples.col(i - 2) = out[0];
        // left shift
        out[0] = out[1];
        out[1] = out[2];
    }
    // reverse pass
    out[0] = samples.col(samples.n_cols - 1);
    out[1] = samples.col(samples.n_cols - 2);
    for (int j = 2; j < samples.n_cols; ++j) {
        int i = samples.n_cols - j - 1;
        out[2] = b0 * samples.col(i) + b1 * samples.col(i + 1) + b2 * samples.col(i + 2) +
                 a1 * out[2 - 1] + a2 * out[2 - 2];
        samples.col(i + 2) = out[0];
        // left shift
        out[0] = out[1];
        out[1] = out[2];
    }
}

void gyro_upsample(arma::mat& samples, int multiplier) {
    if (multiplier < 2) return;
    int length_new = samples.n_cols * multiplier;
    int length = length_new / multiplier;
    int half_mult = multiplier / 2;
    int old_samples_base = length_new - length;
    // std::copy_n(samples, length, samples + old_samples_base);
    samples.cols(old_samples_base, old_samples_base + length - 1) = samples.cols(0, length - 1);

    for (int i = 0; i < length_new; ++i) {
        if ((i + half_mult) % multiplier) {
            samples.col(i) = {};
        } else {
            samples.col(i) = samples.col(i / multiplier + old_samples_base);
        }
    }

    gyro_lowpass(samples, multiplier * 4);
}

void gyro_decimate(arma::mat& samples, int divider) {
    if (divider < 2) return;
    arma::mat samples2 = samples;
    samples.resize(samples.n_rows, samples.n_cols / divider);
    for (int i = 0; i < samples.n_cols / divider; ++i) {
        samples.col(i) = samples2.col(i * divider);
    }
}

int gyro_interpolate(arma::mat& timestamps, arma::mat& gyro) {
    double actual_sr = timestamps.size() / (timestamps.back() - timestamps.front());
    int rounded_sr = int(round(actual_sr / 50) * 50);

    std::vector<double> new_timestamps_vec;
    for (double sample = std::ceil(timestamps.front() * rounded_sr);
         sample / rounded_sr < timestamps.back(); sample += 1)
        new_timestamps_vec.push_back(sample / rounded_sr);

    arma::mat new_timestamps(new_timestamps_vec.data(), 1, new_timestamps_vec.size());
    arma::mat new_gyro(3, new_timestamps_vec.size());
    arma::mat tmp;
    arma::interp1(timestamps, gyro.row(0), new_timestamps, tmp);
    new_gyro.row(0) = tmp;
    arma::interp1(timestamps, gyro.row(1), new_timestamps, tmp);
    new_gyro.row(1) = tmp;
    arma::interp1(timestamps, gyro.row(2), new_timestamps, tmp);
    new_gyro.row(2) = tmp;

    gyro = new_gyro;
    timestamps = new_timestamps;

    return rounded_sr;
}