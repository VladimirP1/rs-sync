#pragma once
#include "public/rssync.h"

#include <ndspline.hpp>

#include <armadillo>

struct FrameData {
    arma::mat ts_a;
    arma::mat ts_b;
    arma::mat rays_a;
    arma::mat rays_b;
};

struct OptData {
    double quats_start{};
    double sample_rate{};
    ndspline quats{};

    // double fps{};
    std::unordered_map<int64_t, FrameData> frame_data{};
};

struct FrameState {
    FrameState(int64_t frame, OptData* problem) : frame_{frame}, problem_{problem} {}

    void Loss(const arma::mat& gyro_delay, const arma::mat& motion_estimate, arma::mat& loss,
              arma::mat& jac_gyro_delay, arma::mat& jac_motion_estimate);
    void Loss(const arma::mat& gyro_delay, const arma::mat& motion_estimate, arma::mat& loss);

    arma::vec3 GuessMotion(double gyro_delay) const;
    double GuessK(double gyro_delay) const;

    arma::mat motion_vec;
    double var_k = 1e3;

   private:
    static constexpr double kNumericDiffStep = 1e-6;

    int64_t frame_;
    OptData* problem_;
};

struct SyncProblemPrivate : public ISyncProblem {
    void SetGyroQuaternions(const double* data, size_t count, double sample_rate,
                            double first_timestamp) override;
    void SetGyroQuaternions(const int64_t* timestamps_us, const double* quats,
                            size_t count) override;
    void SetTrackResult(int64_t frame, const double* ts_a, const double* ts_b, const double* rays_a,
                        const double* rays_b, size_t count) override;
    std::pair<double, double> PreSync(double initial_delay, int64_t frame_begin, int64_t frame_end,
                                      double search_step, double search_radius) override;
    std::pair<double, double> Sync(double initial_delay, int64_t frame_begin, int64_t frame_end,
                                   double search_center, double search_radius) override;

    void DebugPreSync(double initial_delay, int64_t frame_begin, int64_t frame_end,
                      double search_radius, double* delays, double* costs,
                      int point_count) override;

    OptData problem;
};