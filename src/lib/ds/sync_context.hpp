#pragma once

#include <unordered_map>
#include <memory>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <functional>

#include <opencv2/core.hpp>

#include <math/prefix_sums.hpp>
#include <math/range_interpolated.hpp>
#include <math/simple_math.hpp>
#include <vision/calibration.hpp>

class MatchContext {
   private:
    int frame_a_{-1}, frame_b_{-1};
    cv::Mat_<double> points_4d_;
    cv::Mat_<double> projection_;
    // Mosaic corr_maps_;

    std::mutex mtx_{};
};

class FrameContext {
   public:
    FrameContext() {}
    FrameContext(const FrameContext& other) = delete;
    FrameContext& operator=(const FrameContext& other) = delete;

    template <typename T, typename... Args, typename F>
    T InConstContext(F f, Args&&... args) const {
        std::shared_lock<std::shared_timed_mutex> lock{mtx_};
        return f(*this, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args, typename F>
    T InContext(F f, Args&&... args) {
        std::unique_lock<std::shared_timed_mutex> lock{mtx_};
        return f(*this, std::forward<Args>(args)...);
    }

    void SetCorners(std::vector<cv::Point2d> corners, double corner_drop_threshold) {
        corner_drop_threshold_ = corner_drop_threshold;
        corners_ = corners;
    }

    std::vector<cv::Point2d> Corners() const { return corners_; }

    double CornerDropThreshold() const { return corner_drop_threshold_; }

    void SetCachedFrame(cv::Mat frame) {
        cached_frame_ = frame;
    }

    cv::Mat CachedFrame() const {
        return cached_frame_;
    }

   private:
    cv::Mat cached_frame_;
    double corner_drop_threshold_;
    std::vector<cv::Point2d> corners_;
    std::vector<std::shared_ptr<MatchContext>> matched_;

    mutable std::shared_timed_mutex mtx_{};
};

class SyncContext {
   public:
    SyncContext() {}
    SyncContext(const SyncContext& other) = delete;
    SyncContext& operator=(const SyncContext& other) = delete;

    void SetCalibration(const FisheyeCalibration& calibration) {
        std::unique_lock<std::shared_timed_mutex> lock{calibration_mutex_};
        calibration_ = calibration;
    }

    template <typename T, typename... Args, typename F>
    T InFrameContext(int frame, F f, Args&&... args) {
        std::shared_ptr<FrameContext> ctx;
        {
            std::unique_lock<std::mutex> lock{frames_mutex_};
            if (!frames_.count(frame)) {
                frames_.insert({frame, std::make_shared<FrameContext>()});
            }
            ctx = frames_[frame];
        }
        return ctx->InContext<T>(f, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args, typename F>
    T InConstFrameContext(int frame, F f, Args&&... args) {
        std::shared_ptr<FrameContext> ctx;
        {
            std::unique_lock<std::mutex> lock{frames_mutex_};
            if (!frames_.count(frame)) {
                frames_.insert({frame, std::make_shared<FrameContext>()});
            }
            ctx = frames_[frame];
        }
        return ctx->InConstContext<T>(f, std::forward<Args>(args)...);
    }

    template <class T>
    void SetGyroData(T begin, T end) {
        std::unique_lock<std::shared_timed_mutex> lock{gyro_mutex_};
        gyro_integrator_ = {begin, end};
    }

    Quaternion IntegrateGyro(double begin_samples, double end_samples) const {
        std::shared_lock<std::shared_timed_mutex> lock{gyro_mutex_};
        return gyro_integrator_.SoftQuery(begin_samples, end_samples);
    }   

   private:
    FisheyeCalibration calibration_;
    std::unordered_map<int, std::shared_ptr<FrameContext>> frames_;
    Interpolated<PrefixSums<QuaternionGroup>> gyro_integrator_;

    mutable std::shared_timed_mutex calibration_mutex_{};
    mutable std::mutex frames_mutex_{};
    mutable std::shared_timed_mutex gyro_mutex_{};
};