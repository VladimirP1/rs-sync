#include "rough_gyro_correlator.hpp"

#include "pair_storage.hpp"
#include "gyro_loader.hpp"

#include <fstream>
#include <tuple>

#include <opencv2/calib3d.hpp>

namespace rssync {
class RoughGyroCorrelatorImpl : public IRoughGyroCorrelator {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
        gyro_loader_ = ctx_.lock()->GetComponent<IGyroLoader>(kGyroLoaderName);
    }

    void Run(double search_radius, double search_step, int start_frame, int end_frame) override {
        std::vector<FrameInfoT> of_data;

        FillOfData(of_data, start_frame, end_frame);

        double min_cost = std::numeric_limits<double>::max();
        double best_shift = 0.;
        Matrix<double, 3, 1> best_bias;
        for (double shift = -search_radius; shift < search_radius; shift += search_step) {
            auto bias_v = EstimateGyroBias(of_data, shift);

            double cost = RobustCostFunction(of_data, shift, bias_v);

            if (cost / of_data.size() < min_cost) {
                min_cost = cost / of_data.size();
                best_shift = shift;
                best_bias = bias_v;
            }
        }
        std::cout << "Sync: " << best_shift << std::endl;

        ExportSyncPlot(of_data, search_radius, search_step, "out.csv");
        ExportGyroOfTraces(of_data, best_shift, best_bias, "trace.csv");
        ReplaceRotations(best_shift, best_bias);

        // Replace rotations
    }

   private:
    using FrameInfoT = std::tuple<double, double, Quaternion<Jet<double, 3>>>;

    void FillOfData(std::vector<FrameInfoT>& of_data, int start_frame, int end_frame) const {
        std::vector<int> good_frames;
        pair_storage_->GetFramesWith(good_frames, false, false, true, false, false);
        std::sort(good_frames.begin(), good_frames.end());

        cv::Mat_<double> rv;
        PairDescription desc;
        for (auto frame : good_frames) {
            pair_storage_->Get(frame, desc);
            cv::Rodrigues(desc.R, rv);
            of_data.emplace_back(
                desc.timestamp_a, desc.timestamp_b,
                Quaternion<Jet<double, 3>>::FromRotationVector(
                    {Jet<double, 3>{rv(0)}, Jet<double, 3>{rv(1)}, Jet<double, 3>{rv(2)}}));
        }
    }

    Matrix<double, 3, 1> EstimateGyroBias(const std::vector<FrameInfoT>& of_data, double shift,
                                          int iterations = 50, double initial_thresh = 1e-5,
                                          double req_inlier_ratio = 1 / 4.) const {
        std::vector<Matrix<double, 3, 1>> biases;
        for (auto frame_info : of_data) {
            auto of_rot = std::get<2>(frame_info);
            auto gyro_rot = gyro_loader_->GetRotation(std::get<0>(frame_info) + shift,
                                                      std::get<1>(frame_info) + shift);

            auto error = gyro_rot * of_rot.inverse();
            auto bias = GetBiasForOffset(error);
            biases.push_back(bias);
        }

        Matrix<double, 3, 1> bias_v;
        int inliers = 0;
        double thresh = initial_thresh;
        while (inliers < biases.size() * req_inlier_ratio) {
            for (int i = 0; i < iterations; ++i) {
                auto b = biases[static_cast<size_t>(rand()) % biases.size()];
                inliers = 0;
                bias_v = {0, 0, 0};
                for (auto b2 : biases) {
                    double d = (b - b2).norm();
                    if (d < thresh) {
                        ++inliers;
                        bias_v += b2;
                    }
                }
            }
            thresh *= 2;
        }

        // std::cout << thresh << " " << inliers << std::endl;

        bias_v /= inliers;

        return bias_v;
    }

    double RobustCostFunction(const std::vector<FrameInfoT>& of_data, double shift,
                              Matrix<double, 3, 1> bias_v) {
        double cost = 0;
        for (auto frame_info : of_data) {
            auto of_rot = std::get<2>(frame_info);
            auto gyro_rot = gyro_loader_->GetRotation(std::get<0>(frame_info) + shift,
                                                      std::get<1>(frame_info) + shift);

            auto residual =
                (Bias(gyro_rot, bias_v) * Bias(of_rot, {}).inverse()).ToRotationVector().norm();
            cost += log(1. + residual * 100.);
        }
        return cost;
    }

    void ExportSyncPlot(const std::vector<FrameInfoT>& of_data, double search_radius,
                        double search_step, std::string filename) {
        std::ofstream out{filename};

        for (double shift = -search_radius; shift < search_radius; shift += search_step) {
            auto bias_v = EstimateGyroBias(of_data, shift);

            double cost = RobustCostFunction(of_data, shift, bias_v);

            out << shift << "," << cost << std::endl;
        }
    }

    void ExportGyroOfTraces(const std::vector<FrameInfoT>& of_data, double shift,
                            Matrix<double, 3, 1> bias_v, std::string filename) {
        std::ofstream out{filename};

        for (auto frame_info : of_data) {
            double x, y, z;
            auto of_rot = std::get<2>(frame_info);
            auto gyro_rot = gyro_loader_->GetRotation(std::get<0>(frame_info) + shift,
                                                      std::get<1>(frame_info) + shift);
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