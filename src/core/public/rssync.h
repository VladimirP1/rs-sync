#pragma once

#include "export.h"

#include <cstddef>

class ISyncProblem {
   public:
    virtual ~ISyncProblem();

    virtual void SetGyroQuaternions(const double* data, size_t count, double sample_rate,
                                    double first_timestamp) = 0;
    virtual void SetTrackResult(int frame, const double* ts_a, const double* ts_b,
                                const double* rays_a, const double* rays_b, size_t count) = 0;
    virtual void SetFps(double fps) = 0;
    virtual double PreSync(double initial_delay, int frame_begin, int frame_end, double search_step,
                           double search_radius) = 0;
    virtual double Sync(double initial_delay, int frame_begin, int frame_end) = 0;
};

RSSYNC_API ISyncProblem* CreateSyncProblem();