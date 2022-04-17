// The plots of SVD-based cost vs optimization-based still do not match exactly ...

#include <iomanip>
#include <iostream>
#include <vector>

#include "cv.hpp"
#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "loss.hpp"
#include "io.hpp"

#include "inline_utils.hpp"

#include <nlohmann/json.hpp>

static double calc(arma::mat P, arma::mat M, double k) {
    arma::mat r = (P * M) * (k / arma::norm(M));
    arma::mat rho = arma::log1p(r % r);
    return sqrt(arma::accu(arma::sqrt(rho)));
}

static constexpr const char* variants[] = {
    "YxZ", "Xyz", "XZy", "Zxy", "zyX", "yxZ", "ZXY", "zYx", "ZYX", "yXz", "YZX", "XyZ",
    "Yzx", "zXy", "YXz", "xyz", "yZx", "XYZ", "zxy", "xYz", "XYz", "zxY", "zXY", "xZy",
    "zyx", "xyZ", "Yxz", "xzy", "yZX", "yzX", "ZYx", "xYZ", "zYX", "ZxY", "yzx", "xZY",
    "Xzy", "XzY", "YzX", "Zyx", "XZY", "yxz", "xzY", "ZyX", "YXZ", "yXZ", "YZx", "ZXy"};

static constexpr int variants_total = 48;

void find_orient() {
    OptData opt_data;
    Lens lens = lens_load("lens.txt", "xlite4k43");
    track_frames(opt_data.flows, lens, "171836AA.MP4", 30, 30 + 60);

    for (int i = 0; i < sizeof(variants); ++i) {
        optdata_fill_gyro(opt_data, "171836AA.CSV", variants[i]);
        double cost = 0;
        for (auto& [frame, _] : opt_data.flows.data) {
            arma::mat P, M;
            opt_compute_problem(frame, -34, opt_data, P);
            M = opt_guess_translational_motion(P, 20);
            double k = 1 / arma::norm(P * M) * 1e2;
            cost += calc(P, M, k);
        }
        std::cout << variants[i] << " " << cost << std::endl;
    }
}

using json = nlohmann::json;

int main(int argc, char** argv) {
    std::ifstream ifs(argv[1]);
    json j = json::parse(ifs);

    OptData opt_data;
    json input = j["input"];
    json params = j["params"];
    json output = j["output"];

    Lens lens = lens_load(input["lens_profile"]["path"].get<std::string>().c_str(),
                          input["lens_profile"]["name"].get<std::string>().c_str());

    int frame_or_begin = atoi(argv[2]);
    int frame_or_end = atoi(argv[3]);

    track_frames(opt_data.flows, lens, input["video_path"].get<std::string>().c_str(),
                 frame_or_begin, frame_or_end);

    std::vector<std::tuple<double, double, const char*>> results;
    for (int i = 0; i < variants_total; ++i) {
        std::cout << "testing " << variants[i] << "..." << std::endl;
        optdata_fill_gyro(opt_data, input["gyro_path"].get<std::string>().c_str(), variants[i]);

        auto sync = pre_sync(
            opt_data, frame_or_begin, frame_or_end, input["initial_guess"].get<double>(),
            input["simple_presync_radius"].get<double>(), input["simple_presync_step"].get<int>());

        results.emplace_back(sync.first, sync.second, variants[i]);
    }

    std::sort(results.begin(), results.end());

    std::cout << std::endl << "----- Top-5 results -----" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << std::get<2>(results[i]) << " " << std::get<0>(results[i]) << std::endl;
    }
}
