#include "rough_gyro_correlator.hpp"

#include "pair_storage.hpp"
#include "gyro_loader.hpp"

#include <math/gyro_integrator.hpp>

#include <fstream>
#include <tuple>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <opencv2/calib3d.hpp>

using Eigen::Matrix;

namespace rssync {
class RoughGyroCorrelatorImpl : public IRoughGyroCorrelator {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);

        std::vector<Eigen::Vector3d> gyro_data(gyro_loader_->DataSize());
        gyro_loader_->GetData(gyro_data.data(), gyro_data.size());

        sample_rate_ = gyro_loader_->SampleRate();
        LowpassGyro(gyro_data.data(), gyro_data.size(), sample_rate_ / 30.);

        integrator_ = {gyro_data.data(), static_cast<int>(gyro_data.size())};
    }

    void Run(double initial_offset, double search_radius, double search_step, int start_frame,
             int end_frame, RoughCorrelationReport* report) override {
        std::vector<FrameInfoT> of_data;

        FillOfData(of_data, start_frame, end_frame);

        std::vector<int> inliers, best_inliers;
        double min_cost = std::numeric_limits<double>::max();
        double best_shift = 0.;
        Matrix<double, 3, 1> best_bias;
        for (double shift = initial_offset - search_radius; shift < initial_offset + search_radius;
             shift += search_step) {
            inliers.clear();
            Matrix<double, 3, 1> bias_v;
            double cost = RobustCostFunction(of_data, inliers, shift, bias_v);

            if (cost < min_cost) {
                min_cost = cost;
                best_shift = shift;
                best_bias = bias_v;
                best_inliers = inliers;
            }
        }
        std::cout << "Sync: " << best_shift << std::endl;

        if (report) {
            report->offset = best_shift;
            report->bias_estimate = best_bias;
            for (auto i : best_inliers) {
                report->frames.push_back(std::get<0>(of_data[i]));
            }
        }

        ExportSyncPlot(of_data, initial_offset, search_radius, search_step, "out.csv");
        ExportGyroOfTraces2(of_data, best_shift, best_bias, "trace.csv");
        // ReplaceRotations(best_shift, best_bias);
    }

   private:
    using FrameInfoT = std::tuple<int, double, double, Eigen::Vector3d>;

    template <class T>
    Eigen::AngleAxis<T> ToAngleAxis(const Eigen::Matrix<T, 3, 1>& v) const {
        return {v.norm(), v.normalized()};
    }

    void FillOfData(std::vector<FrameInfoT>& of_data, int start_frame, int end_frame) const {
        std::vector<int> good_frames;
        pair_storage_->GetFramesWith(good_frames, false, false, true, false, false);
        std::sort(good_frames.begin(), good_frames.end());

        cv::Mat_<double> rv;
        PairDescription desc;
        for (auto frame : good_frames) {
            if (frame >= end_frame || frame < start_frame) continue;
            pair_storage_->Get(frame, desc);
            cv::Rodrigues(desc.R, rv);
            of_data.emplace_back(frame, desc.timestamp_a, desc.timestamp_b,
                                 Eigen::Vector3d{rv(0), rv(1), rv(2)});
        }
    }

    double RobustCostFunction(const std::vector<FrameInfoT>& of_data, std::vector<int>& inliers,
                              double shift, Eigen::Matrix<double, 3, 1>& bias, int iterations = 20,
                              double req_inlier_ratio = 1 / 2.) const {
        std::vector<Eigen::Matrix<double, 3, 1>> biases;
        for (auto frame_info : of_data) {
            auto of_rot = std::get<3>(frame_info);
            auto gyro = integrator_.IntegrateGyro((std::get<1>(frame_info) + shift) * sample_rate_,
                                                  (std::get<2>(frame_info) + shift) * sample_rate_);
            biases.push_back(gyro.FindBias(of_rot));
        }

        Eigen::Vector3d base_inlier{};
        double thresh = std::numeric_limits<double>::max();
        std::vector<double> norms;
        // Estimate the inlier threshold
        for (int i = 0; i < iterations; ++i) {
            auto b0 = biases[static_cast<size_t>(rand()) % biases.size()];

            norms.clear();
            for (auto& b1 : biases) {
                norms.push_back((b0 - b1).norm());
            }
            auto nth_it = norms.begin() + req_inlier_ratio * norms.size();

            std::nth_element(norms.begin(), nth_it, norms.end());

            if (2 * *nth_it < thresh) {
                thresh = 2 * *nth_it;
                base_inlier = b0;
            }
        }
        // std::cout << thresh << "\n" << base_inlier << "\n----------\n" << std::endl;

        // Main part
        inliers.clear();
        double best_cost = 1;
        Eigen::Vector3d best_bias{};

        bias = {0, 0, 0};
        inliers.clear();
        for (int ib1 = 0; ib1 < biases.size(); ++ib1) {
            const auto& b1 = biases[ib1];
            if ((b1 - base_inlier).norm() < thresh) {
                bias += b1;
                inliers.push_back(ib1);
            }
        }
        bias /= inliers.size();

        // Calculate cost
        double cost = 0;
        double count = 0;
        for (auto j : inliers) {
            const auto& frame_info = of_data[j];
            auto of_rot = std::get<3>(frame_info);
            auto gyro = integrator_.IntegrateGyro((std::get<1>(frame_info) + shift) * sample_rate_,
                                                  (std::get<2>(frame_info) + shift) * sample_rate_);

            auto residual = Eigen::AngleAxis<double>(ToAngleAxis(gyro.Bias(bias).rot) *
                                                     ToAngleAxis(of_rot).inverse())
                                .angle();

            cost += log(1. + residual * 100.);
            count += 1;
        }
        cost /= count;

        if (cost < best_cost) {
            best_cost = cost;
            best_bias = bias;
        }

        bias = best_bias;
        return best_cost;
    }

    void ExportSyncPlot(const std::vector<FrameInfoT>& of_data, double initial_offset,
                        double search_radius, double search_step, std::string filename) {
        std::ofstream out{filename};

        for (double shift = initial_offset - search_radius; shift < initial_offset + search_radius;
             shift += search_step) {
            Matrix<double, 3, 1> bias;
            std::vector<int> inliers;
            double cost = RobustCostFunction(of_data, inliers, shift, bias);

            out << shift << "," << cost << "\n";
        }
    }

    void ExportGyroOfTraces(const std::vector<FrameInfoT>& of_data, double shift,
                            Matrix<double, 3, 1> bias_v, std::string filename) {
        std::ofstream out{filename};

        for (auto frame_info : of_data) {
            auto of_rot = std::get<3>(frame_info);
            auto gyro = integrator_.IntegrateGyro((std::get<1>(frame_info) + shift) * sample_rate_,
                                                  (std::get<2>(frame_info) + shift) * sample_rate_);
            auto rv_gyro = gyro.Bias(bias_v).rot;

            out << of_rot.x() << "," << of_rot.y() << "," << of_rot.z() << "," << rv_gyro.x() << ","
                << rv_gyro.y() << "," << rv_gyro.z() << std::endl;
        }
    }

    void ExportGyroOfTraces2(const std::vector<FrameInfoT>& of_data, double shift,
                             Matrix<double, 3, 1> bias_v, std::string filename) {
        std::ofstream out{filename};
        out << std::fixed << std::setprecision(16);

        std::vector<Eigen::Vector3d> of_gyr;
        for (auto frame_info : of_data) {
            auto of_rot = std::get<3>(frame_info);
            of_gyr.push_back(of_rot);
        }
        double frame_rate = 1. / (std::get<2>(of_data.front()) - std::get<1>(of_data.front())),
               actual_frame_rate = frame_rate;
        std::cout << frame_rate << std::endl;
        of_gyr.resize(of_gyr.size() * 34);
        UpsampleGyro(of_gyr.data(), of_gyr.size(), 34);
        frame_rate *= 34;
        LowpassGyro(of_gyr.data(), of_gyr.size(), frame_rate / 1);
        GyroIntegrator of_int(of_gyr.data(), of_gyr.size());

        std::vector<Eigen::Vector3d> gyro_data(gyro_loader_->DataSize());
        gyro_loader_->GetData(gyro_data.data(), gyro_data.size());
        LowpassGyro(gyro_data.data(), gyro_data.size(), sample_rate_ / 1);
        GyroIntegrator gyro_int(gyro_data.data(), gyro_data.size());

        double step = .01;
        double vid_start = std::get<1>(of_data.front()) + 1. / actual_frame_rate;
        double vid_end = std::get<1>(of_data.back());

        for (double ts = vid_start; ts < vid_end; ts += step) {
            auto gyro = gyro_int.IntegrateGyro((ts + shift) * sample_rate_,
                                               (ts + step + shift) * sample_rate_);
            auto of = of_int.IntegrateGyro((ts - vid_start + shift) * frame_rate,
                                           (ts - vid_start + step + shift) * frame_rate);
            auto rv_gyro = gyro.Bias(bias_v).rot;

            out << of.rot.x() << "," << of.rot.y() << "," << of.rot.z() << "," << rv_gyro.x() << ","
                << rv_gyro.y() << "," << rv_gyro.z() << std::endl;
        }
    }

    // void ReplaceRotations(double best_shift, Matrix<double, 3, 1> best_bias) {
    //     std::vector<int> good_frames;
    //     pair_storage_->GetFramesWith(good_frames, false, false, true, false, false);
    //     std::sort(good_frames.begin(), good_frames.end());
    //     PairDescription desc;

    //     for (auto frame : good_frames) {
    //         pair_storage_->Get(frame, desc);
    //         auto gyro_rot = gyro_loader_->GetRotation(desc.timestamp_a + best_shift,
    //                                                   desc.timestamp_b + best_shift);
    //         auto rve = Bias(gyro_rot, best_bias).ToRotationVector();
    //         cv::Mat_<double> rv(3, 1, CV_64F);
    //         rv << rve.x(), rve.y(), rve.z();
    //         cv::Rodrigues(rv, desc.R);
    //         // desc.t << 0,0,0;
    //         pair_storage_->Update(frame, desc);
    //     }

    //     std::cout << "Rotations updated" << std::endl;
    // }

    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
    GyroIntegrator integrator_;
    double sample_rate_;
};

void RegisterRoughGyroCorrelator(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<RoughGyroCorrelatorImpl>(ctx, name);
}

}  // namespace rssync