#include <execution>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>
#include <fstream>

#include "cv.hpp"
#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "io.hpp"
#include "loss.hpp"
#include "backtrack.hpp"

#include <nlohmann/json.hpp>

struct FrameState {
    FrameState(int frame, OptData* optdata) : frame_{frame}, optdata_{optdata} {}

    bool Cost(const arma::mat& gyro_delay, arma::mat& cost, arma::mat& jac_gyro_delay) const {
        arma::mat P, PL, PR;

        opt_compute_problem(frame_, gyro_delay[0] - kStep, *optdata_, PL);
        opt_compute_problem(frame_, gyro_delay[0] + kStep, *optdata_, PR);

        double r1 = calc(PL, var_k);
        double r2 = calc(PR, var_k);

        cost = (r1 + r2) / 2;

        jac_gyro_delay = (r2 - r1) / 2 / kStep;

        return true;
    }

    bool CostOnly(const arma::mat& gyro_delay, arma::mat& cost) const {
        arma::mat P;
        opt_compute_problem(frame_, gyro_delay[0], *optdata_, P);
        cost = {calc(P, var_k)};
        return true;
    }

    double GuessK(double gyro_delay) const {
        arma::mat problem;
        opt_compute_problem(frame_, gyro_delay, *optdata_, problem);
        return 1 / arma::norm(problem) * 1e2;
    }

    arma::mat gyro_delay;

    double var_k = 1e3;
    
   private:
    static constexpr double kStep = 1e-6;

    int frame_;
    OptData* optdata_;

    static double calc(arma::mat P, double k) {
        return arma::accu(arma::log1p(arma::sum(P % P, 1) * k * k));
    }
};

struct backtrack_hyper {
    double c{.7};
    double tau{.1};
    double limit{20};
    double step_init{1};
};

struct opt_result {
    double delay;
};

opt_result opt_run(OptData& data, double initial_delay,
                   int min_frame = std::numeric_limits<int>::min(),
                   int max_frame = std::numeric_limits<int>::max()) {
    arma::mat gyro_delay(1, 1);
    gyro_delay[0] = initial_delay;

    Backtrack delay_opt;
    delay_opt.SetHyper(.2, .1, 1e-2, 20);

    std::vector<std::unique_ptr<FrameState>> costs;
    for (auto& [frame, _] : data.flows.data) {
        if (frame < min_frame || frame > max_frame) continue;
        costs.push_back(std::make_unique<FrameState>(frame, &data));
        costs.back()->var_k = costs.back()->GuessK(gyro_delay[0]);
    }

    struct delay_opt_info {
        double step_size;
    };

    constexpr double delay_b{.3};
    arma::mat delay_v(1, 1);

    delay_opt.SetObjective([&](arma::vec x) {
        std::mutex m;
        arma::mat cost(1, 1), delay_g(1, 1);
        std::for_each(std::execution::par, costs.begin(), costs.end(),
                      [x, &cost, &delay_g, &m](auto& fs) {
                          arma::mat cur_cost, cur_delay_g, tmp;
                          fs->Cost(x, cur_cost, cur_delay_g);
                          std::unique_lock<std::mutex> lock(m);
                          cost += cur_cost;
                          delay_g += cur_delay_g;
                      });
        return std::make_pair(cost[0], delay_g);
    });

    delay_opt.SetObjective([&](arma::vec x) {
        std::mutex m;
        arma::mat cost(1, 1);
        std::for_each(std::execution::par, costs.begin(), costs.end(), [x, &cost, &m](auto& fs) {
            arma::mat cur_cost;
            fs->CostOnly(x, cur_cost);
            std::unique_lock<std::mutex> lock(m);
            cost += cur_cost;
        });
        return cost[0];
    });

    auto do_opt_delay = [&]() {
        arma::mat bt = delay_opt.Step(gyro_delay - delay_b * delay_v);

        delay_v = delay_b * delay_v + bt;
        gyro_delay += delay_v;

        return delay_opt_info{arma::norm(bt)};
    };

    int converge_counter = 0;

    for (int i = 0; i < 1000; i++) {
        // Optimize delay
        auto info = do_opt_delay();

        if (info.step_size < 1e-6) {
            converge_counter++;
        } else {
            converge_counter = 0;
        }

        if (converge_counter > 5) {
            // std::cout << "Converged at " << i << std::endl;
            break;
        }

        std::cerr << gyro_delay[0] << " " << info.step_size << std::endl;
    }

    return {gyro_delay[0]};
}

using json = nlohmann::json;

int main(int argc, char** argv) {
    std::ifstream ifs(argv[1]);
    json j = json::parse(ifs);

    OptData opt_data;
    json input = j["input"];
    json params = j["params"];
    json output = j["output"];
    optdata_fill_gyro(opt_data, input["gyro_path"].get<std::string>().c_str(),
                      input["gyro_orientation"].get<std::string>().c_str());

    Lens lens = lens_load(input["lens_profile"]["path"].get<std::string>().c_str(),
                          input["lens_profile"]["name"].get<std::string>().c_str());

    int frame_start = input["frame_range"][0].get<int>();
    int frame_end = input["frame_range"][1].get<int>();
    int sync_window = params["sync_window"].get<int>();
    int syncpoint_distance = params["syncpoint_distance"].get<int>();
    track_frames(opt_data.flows, lens, input["video_path"].get<std::string>().c_str(),
                 input["frame_range"][0].get<int>(), input["frame_range"][1].get<int>());

    std::vector<int> syncpoints;
    if (params["syncpoints_format"] == "auto") {
        for (int pos = frame_start; pos + syncpoint_distance < frame_end; pos += syncpoint_distance) 
            syncpoints.push_back(pos);
    } else if (params["syncpoints_format"] == "array") {
        for (int pos : params["syncpoints_array"]) {
            syncpoints.push_back(pos);
        }
    } else {
        return 1;
    }

    std::ofstream csv("_" + output["csv_path"].get<std::string>());

    for (auto pos : syncpoints) {
        std::cerr << pos << std::endl;
        double delay = input["initial_guess"].get<double>();
        for (int i = 0; i < 4; ++i) delay = opt_run(opt_data, delay, pos, pos + sync_window).delay;
        csv << pos << "," << delay << std::endl;
    }

    csv.close();

    return 0;
}