#include "visualizer.hpp"

#include <cassert>
#include <numeric>
#include <iostream>

#include <opencv2/imgproc.hpp>

#include "pair_storage.hpp"

namespace rssync {
class VisualizerImpl : public IVisualizer {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {
        pair_storage_ = ctx_.lock()->GetComponent<IPairStorage>(kPairStorageName);
    }

    void DimImage(cv::Mat& frame, double k) override { frame /= (1. / k); }

    void OverlayMatched(cv::Mat& frame, int frame_number, bool ab, bool undistorted) override {
        PairDescription desc;
        if (!pair_storage_->Get(frame_number, desc) || (!desc.has_points && !undistorted) ||
            (!desc.has_undistorted && undistorted)) {
            return;
        }

        std::vector<long>& ids = desc.point_ids;
        std::vector<cv::Point2f>& pts =
            ab ? (undistorted ? desc.points_undistorted_a : desc.points_a)
               : (undistorted ? desc.points_undistorted_b : desc.points_b);

        for (int i = 0; i < ids.size(); ++i) {
            cv::circle(frame, pts[i], 5, GetColor(ids[i]), 3);
            if (desc.has_pose && desc.mask_essential[i]) {
                cv::circle(frame, pts[i], 12, cv::Scalar(0, 0, 255), 4);
            }
        }
    }

    void OverlayMatchedTracks(cv::Mat& frame, int frame_number, bool undistorted) override {
        PairDescription desc;
        if (!pair_storage_->Get(frame_number, desc) || (!desc.has_points && !undistorted) ||
            (!desc.has_undistorted && undistorted)) {
            return;
        }

        std::vector<long>& ids = desc.point_ids;
        std::vector<cv::Point2f>& pts_a = undistorted ? desc.points_undistorted_a : desc.points_a;
        std::vector<cv::Point2f>& pts_b = undistorted ? desc.points_undistorted_b : desc.points_b;

        for (int i = 0; i < ids.size(); ++i) {
            cv::line(frame, pts_a[i], pts_b[i], GetColor(ids[i]), 2);
        }
    }

    bool VisualizeCorrelations(cv::Mat& out, int frame_number, double target_aspect) override {
        PairDescription desc;
        if (!pair_storage_->Get(frame_number, desc) || !desc.has_correlations) {
            return false;
        }

        int target_w{}, target_h{};
        for (int i = 0; i < desc.correlation_models.size(); ++i) {
            if (desc.mask_correlation[i]) {
                if (desc.debug_correlations.size()) {
                    cv::Mat grad_col;
                    CorrelationToColor(desc.debug_correlations[i], grad_col,
                                       cv::COLORMAP_DEEPGREEN);
                    auto corr_size = grad_col.size();
                    auto patch_size_a = desc.debug_patches.empty()
                                            ? cv::Size(0, 0)
                                            : desc.debug_patches[i].first.size();
                    auto patch_size_b = desc.debug_patches.empty()
                                            ? cv::Size(0, 0)
                                            : desc.debug_patches[i].first.size();
                    target_w =
                        std::max(corr_size.width, std::max(patch_size_a.width, patch_size_b.width));
                    target_h = std::max(corr_size.height,
                                        std::max(patch_size_a.height, patch_size_b.height));
                } else {
                    int model_size = desc.corr_valid_radius * 2 * 6;
                    target_w = target_h = model_size;
                }
                break;
            }
        }

        if (target_w == 0 || target_h == 0) return false;

        int total_tiles =
            std::accumulate(desc.mask_correlation.begin(), desc.mask_correlation.end(), 0);

        size_t canvas_w{1}, canvas_h{1};
        while (canvas_w * canvas_h < total_tiles) {
            if ((canvas_w * target_w * 3.) / (canvas_h * target_h * 2.) < target_aspect) {
                ++canvas_w;
            } else {
                ++canvas_h;
            }
        }

        out = cv::Mat::zeros(canvas_h * target_h * 2, canvas_w * target_w * 3, CV_8UC3);

        int k{};
        for (int i = 0; i < canvas_w; i += 1) {
            for (int j = 0; j < canvas_h; j += 1) {
                while (!desc.mask_correlation[k] && k < desc.mask_correlation.size()) ++k;
                if (k >= desc.mask_correlation.size()) break;
                auto base_row = j * target_h * 2;
                auto base_col = i * target_w * 3;
                auto corr_roi = cv::Rect(base_col, base_row, target_w, target_h);
                auto corr_model_roi = cv::Rect(base_col + target_w, base_row, target_w, target_h);
                auto none0_roi = cv::Rect(base_col + target_w * 2, base_row, target_w, target_h);
                auto none1_roi =
                    cv::Rect(base_col + target_w * 2, base_row + target_h, target_w, target_h);
                auto a_roi = cv::Rect(base_col, base_row + target_h, target_w, target_h);
                auto b_roi = cv::Rect(base_col + target_w, base_row + target_h, target_w, target_h);

                cv::Mat grad_col, tmp;
                double tx, ty;
                
                NormalModel model = desc.correlation_models[k];
                model.ShiftOrigin(desc.corr_valid_radius, desc.corr_valid_radius);
                model.GetCenter(tx, ty);

                if (!desc.debug_patches.empty()) {
                    cv::Mat_<double> affine(2, 3, CV_64F);
                    cv::Size src_size = desc.debug_patches[k].first.size();
                    affine << 2, 0, (a_roi.height - src_size.height) / 2., 0, 2.,
                        (a_roi.width - src_size.width) / 2.;
                    cv::warpAffine(desc.debug_patches[k].first, out(a_roi), affine, a_roi.size());

                    src_size = desc.debug_patches[k].second.size();
                    affine << 2, 0, (a_roi.height - src_size.height) / 2., 0, 2.,
                        (a_roi.width - src_size.width) / 2.;
                    cv::warpAffine(desc.debug_patches[k].second, out(b_roi), affine, b_roi.size());

                    CorrelationToColor(desc.debug_correlations[k], grad_col, cv::COLORMAP_MAGMA);
                    cv::circle(grad_col, cv::Point((tx + .5) * 6, (ty + .5) * 6), 1,
                               cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    cv::resize(grad_col, out(corr_roi), corr_roi.size());
                }

                ComputeNormalImage(
                    tmp, model,
                    cv::Size(desc.corr_valid_radius * 2, desc.corr_valid_radius * 2));
                CorrelationToColor(tmp, grad_col, cv::COLORMAP_MAGMA);
                cv::circle(grad_col, cv::Point((tx + .5) * 6, (ty + .5) * 6), 1,
                           cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::resize(grad_col, out(corr_model_roi), corr_model_roi.size());

                ++k;
            }
        }

        return true;
    }

   private:
    cv::Scalar GetColor(long id) {
        size_t idx = static_cast<size_t>(id) % kColorPalleteSize;
        return cv::Scalar(kColorPallete[idx][2], kColorPallete[idx][1], kColorPallete[idx][0]);
    }

    static constexpr size_t kColorPalleteSize = 10;
    static constexpr uchar kColorPallete[10][3] = {
        {249, 65, 68},   {243, 114, 44}, {248, 150, 30}, {249, 132, 74}, {249, 199, 79},
        {144, 190, 109}, {67, 170, 139}, {77, 144, 142}, {87, 117, 144}, {39, 125, 161}};

    void ComputeNormalImage(cv::Mat& out, NormalModel model, cv::Size size) {
        out.create(size, CV_32FC1);
        for (int i = 0; i < out.rows; ++i) {
            for (int j = 0; j < out.cols; ++j) {
                out.at<float>(i, j) = model.Evaluate(j, i);
            }
        }
    }

    void CorrelationGradToColor(const cv::Mat& correlation, cv::Mat& colorized) {
        colorized = cv::Mat::zeros(correlation.rows, correlation.cols, CV_8UC3);
        for (int i = 0; i < correlation.rows; ++i) {
            for (int j = 0; j < correlation.cols; ++j) {
                auto grad = correlation.at<cv::Vec2f>(i, j);
                auto angle = atan2(grad[1], grad[0]);
                uchar hue = angle * 255. / M_PI;
                colorized.at<cv::Vec3b>(i, j) = {hue, 255, 127};
                // std::cout <<(int) hue << std::endl;
            }
        }
        cv::cvtColor(colorized, colorized, cv::COLOR_HSV2BGR);
    }

    void CorrelationToColor(const cv::Mat& correlation, cv::Mat& colorized, cv::ColormapTypes t) {
        auto ucorr = correlation.clone();
        // ucorr = cv::abs(ucorr);
        double min, max;
        cv::minMaxLoc(ucorr, &min, &max);
        ucorr -= min;
        // ucorr /= (max - min);
        ucorr.convertTo(ucorr, CV_8UC1, 255);
        cv::cvtColor(ucorr, ucorr, cv::COLOR_GRAY2BGR);
        cv::resize(ucorr, ucorr, ucorr.size() * 6, 0, 0, cv::INTER_CUBIC);
        cv::applyColorMap(ucorr, ucorr, t);
        cv::circle(ucorr, {ucorr.cols / 2, ucorr.rows / 2}, 1, cv::Scalar(0, 255, 0), 1,
                   cv::LINE_AA);
        colorized = ucorr;
    }

   private:
    std::shared_ptr<IPairStorage> pair_storage_;
};

void RegisterVisualizer(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<VisualizerImpl>(ctx, name);
}
}  // namespace rssync