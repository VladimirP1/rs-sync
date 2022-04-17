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

    int frame_begin = input["frame_range"][0].get<int>();
    int frame_end = input["frame_range"][1].get<int>();

    auto start_ts = std::chrono::steady_clock::now();
    track_frames(opt_data.flows, lens, input["video_path"].get<std::string>().c_str(),
                 frame_begin, frame_end);
    auto stop_ts = std::chrono::steady_clock::now();

    std::cout << "avg fps = " << (frame_end - frame_begin) * 1e3 / std::chrono::duration_cast<std::chrono::milliseconds>(stop_ts - start_ts).count() << std::endl;
}
