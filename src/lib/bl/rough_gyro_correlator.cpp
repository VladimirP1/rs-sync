#include "rough_gyro_correlator.hpp"

#include "pair_storage.hpp"
#include "gyro_loader.hpp"

#include <fstream>
#include <tuple>
#include <cmath>

#include <opencv2/calib3d.hpp>

namespace rssync {
class RoughGyroCorrelatorImpl : public IRoughGyroCorrelator {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
    }

    void Run(double initial_offset, double search_radius, double search_step, int start_frame,
             int end_frame, RoughCorrelationReport* report) override {
        std::vector<FrameInfoT> of_data;

        FillOfData(of_data, start_frame, end_frame);

        double min_cost = std::numeric_limits<double>::max();
        double best_shift = 0.;
        Matrix<double, 3, 1> best_bias;
        for (double shift = initial_offset - search_radius; shift < initial_offset + search_radius;
             shift += search_step) {
            Matrix<double, 3, 1> bias_v;

            double cost = RobustCostFunction(of_data, shift, bias_v);

            if (cost < min_cost) {
                min_cost = cost;
                best_shift = shift;
                best_bias = bias_v;
            }
        }
        std::cout << "Sync: " << best_shift << std::endl;

        if (report) {
            report->offset = best_shift;
            report->bias_estimate = best_bias;
        }

        ExportSyncPlot(of_data, initial_offset, search_radius, search_step, "out.csv");
        ExportGyroOfTraces(of_data, best_shift, best_bias, "trace.csv");
        // ReplaceRotations(best_shift, best_bias);

        // Replace rotations
    }

   private:
    using FrameInfoT = std::tuple<int, double, double, Quaternion<Jet<double, 3>>>;

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
            of_data.emplace_back(
                frame, desc.timestamp_a, desc.timestamp_b,
                Quaternion<Jet<double, 3>>::FromRotationVector(
                    {Jet<double, 3>{rv(0)}, Jet<double, 3>{rv(1)}, Jet<double, 3>{rv(2)}}));
        }
    }

    double RobustCostFunction(const std::vector<FrameInfoT>& of_data, double shift,
                              Matrix<double, 3, 1>& bias, int iterations = 10,
                              double initial_thresh = 1e-5,
                              double req_inlier_ratio = 1 / 2.) const {
        std::vector<Matrix<double, 3, 1>> biases;
        for (auto frame_info : of_data) {
            auto of_rot = std::get<3>(frame_info);
            auto gyro_rot = gyro_loader_->GetRotation(std::get<1>(frame_info) + shift,
                                                      std::get<2>(frame_info) + shift);

            auto error = gyro_rot * of_rot.inverse();
            auto bias = GetBiasForOffset(error);
            biases.push_back(bias);
        }

        Matrix<double, 3, 1> bias_v{};
        int inliers = 0;
        double thresh = initial_thresh;
        // Estimate the RANSAC threshold
        while (true) {
            for (int i = 0; i < iterations; ++i) {
                auto b0 = biases[static_cast<size_t>(rand()) % biases.size()];
                inliers = 0;
                for (const auto& b1 : biases) {
                    auto diff = (b1 - b0).norm();
                    if (diff < thresh) {
                        ++inliers;
                    }
                }
                if (inliers >= biases.size() * req_inlier_ratio) {
                    break;
                }
            }
            if (inliers >= biases.size() * req_inlier_ratio) {
                break;
            }
            thresh *= 2;
        }

        double best_cost = 1;
        Matrix<double, 3, 1> best_bias{};
        std::vector<int> inliers_idx;
        // Main RANSAC
        for (int i = 0; i < iterations; ++i) {
            auto b0 = biases[static_cast<size_t>(rand()) % biases.size()];
            bias_v = {0, 0, 0};
            inliers = 0;
            inliers_idx.clear();
            for (int ib1 = 0; ib1 < biases.size(); ++ib1) {
                const auto& b1 = biases[ib1];
                if ((b1 - b0).norm() < thresh) {
                    ++inliers;
                    bias_v += b1;
                    inliers_idx.push_back(ib1);
                }
            }
            bias_v /= inliers;

            if (inliers < biases.size() * req_inlier_ratio) {
                continue;
            }

            // Calculate cost
            double cost = 0;
            double count = 0;
            for (auto j : inliers_idx) {
                const auto& frame_info = of_data[j];
                auto of_rot = std::get<3>(frame_info);
                auto gyro_rot = gyro_loader_->GetRotation(std::get<1>(frame_info) + shift,
                                                          std::get<2>(frame_info) + shift);

                auto residual =
                    (Bias(gyro_rot, bias_v) * Bias(of_rot, {}).inverse()).ToRotationVector().norm();
                cost += log(1. + residual * 100.);
                count += 1;
            }
            cost /= count;

            if (cost < best_cost) {
                best_cost = cost;
                best_bias = bias_v;
            }
        }

        bias = best_bias;
        return best_cost;
    }

    void ExportSyncPlot(const std::vector<FrameInfoT>& of_data, double initial_offset,
                        double search_radius, double search_step, std::string filename) {
        std::ofstream out{filename};

        for (double shift = initial_offset - search_radius; shift < initial_offset + search_radius;
             shift += search_step) {
            // auto bias_v = EstimateGyroBias(of_data, inlier_data, shift);
            Matrix<double, 3, 1> bias;

            double cost = RobustCostFunction(of_data, shift, bias);

            out << shift << "," << cost << std::endl;
        }
    }

    void ExportGyroOfTraces(const std::vector<FrameInfoT>& of_data, double shift,
                            Matrix<double, 3, 1> bias_v, std::string filename) {
        std::ofstream out{filename};

        for (auto frame_info : of_data) {
            double x, y, z;
            auto of_rot = std::get<3>(frame_info);
            auto gyro_rot = gyro_loader_->GetRotation(std::get<1>(frame_info) + shift,
                                                      std::get<2>(frame_info) + shift);
            auto rv_of = Bias(of_rot, {}).ToRotationVector();
            auto rv_gyro = Bias(gyro_rot, bias_v).ToRotationVector();

            out << rv_of.x() << "," << rv_of.y() << "," << rv_of.z() << "," << rv_gyro.x() << ","
                << rv_gyro.y() << "," << rv_gyro.z() << std::endl;
        }
    }

    void ReplaceRotations(double best_shift, Matrix<double, 3, 1> best_bias) {
        std::vector<int> good_frames;
        pair_storage_->GetFramesWith(good_frames, false, false, true, false, false);
        std::sort(good_frames.begin(), good_frames.end());
        PairDescription desc;

        for (auto frame : good_frames) {
            pair_storage_->Get(frame, desc);
            auto gyro_rot = gyro_loader_->GetRotation(desc.timestamp_a + best_shift,
                                                      desc.timestamp_b + best_shift);
            auto rve = Bias(gyro_rot, best_bias).ToRotationVector();
            cv::Mat_<double> rv(3, 1, CV_64F);
            rv << rve.x(), rve.y(), rve.z();
            cv::Rodrigues(rv, desc.R);
            // desc.t << 0,0,0;
            pair_storage_->Update(frame, desc);
        }

        std::cout << "Rotations updated" << std::endl;
    }

    std::shared_ptr<IPairStorage> pair_storage_;
    std::shared_ptr<IGyroLoader> gyro_loader_;
};

void RegisterRoughGyroCorrelator(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<RoughGyroCorrelatorImpl>(ctx, name);
}

}  // namespace rssync