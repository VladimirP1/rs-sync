#pragma once

#include "export.h"

#include <cstddef>
#include <cstdint>
#include <utility>

class ISyncProblem {
   public:
    virtual ~ISyncProblem();

    virtual void SetGyroQuaternions(const double* data, size_t count, double sample_rate,
                                    double first_timestamp) = 0;
    virtual void SetGyroQuaternions(const uint64_t* timestamps_us, const double* quats,
                                    size_t count) = 0;
    virtual void SetTrackResult(int frame, const double* ts_a, const double* ts_b,
                                const double* rays_a, const double* rays_b, size_t count) = 0;
    virtual std::pair<double, double> PreSync(double initial_delay, int frame_begin, int frame_end,
                                              double search_step, double search_radius) = 0;
    virtual std::pair<double, double> Sync(double initial_delay, int frame_begin,
                                           int frame_end) = 0;

    virtual void DebugPreSync(double initial_delay, int frame_begin, int frame_end,
                              double search_radius, double* delays, double* costs,
                              int point_count) = 0;
};

RSSYNC_API ISyncProblem* CreateSyncProblem();