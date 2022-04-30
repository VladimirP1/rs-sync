#include "core_private.hpp"

#include <inline_utils.hpp>
#include <quat.hpp>
#include <backtrack.hpp>

#include <ensmallen_bits/log.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include <ensmallen_bits/utility/arma_traits.hpp>
#include <ensmallen_bits/lbfgs/lbfgs.hpp>

#include <execution>

arma::mat opt_compute_problem(int frame, double gyro_delay, const OptData& data) {
    const auto& flow = data.frame_data.at(frame);
    arma::mat ap = flow.rays_a;
    arma::mat bp = flow.rays_b;
    double baset = ((frame / data.fps) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat at = ((flow.ts_a) - data.quats_start + gyro_delay) * data.sample_rate;
    arma::mat bt = (flow.ts_b - data.quats_start + gyro_delay) * data.sample_rate;

    arma::mat problem(at.n_cols, 3);
    arma::vec4 base_conj = quat_conj(data.quats.eval(baset));
    for (int i = 0; i < at.n_cols; ++i) {
        arma::vec4 a = data.quats.eval(at[i]);
        arma::vec4 b = data.quats.eval(bt[i]);
        double inv_base_a_norm = (1. / arma::norm(base_conj)) * (1. / arma::norm(a));
        double inv_base_b_norm = (1. / arma::norm(base_conj)) * (1. / arma::norm(b));
        arma::vec4 rot_a = quat_prod(base_conj, a) * inv_base_a_norm;
        arma::vec4 rot_b = quat_prod(base_conj, b) * inv_base_b_norm;
        arma::vec3 ar = quat_rotate_point(quat_conj(rot_a), ap.col(i));
        arma::vec3 br = quat_rotate_point(quat_conj(rot_b), bp.col(i));
        problem.row(i) = arma::trans(arma::cross(ar, br));
    }

    return problem;
}

arma::vec3 opt_guess_translational_motion(const arma::mat& problem, int max_iters) {
    arma::mat nproblem = problem;
    nproblem.each_row([](arma::mat& m) { m = safe_normalize(m); });

    arma::vec3 best_sol;
    double least_med = std::numeric_limits<double>::infinity();
    for (int i = 0; i < max_iters; ++i) {
        int vs[2];
        vs[0] = vs[1] = mtrand(0, problem.n_rows - 1);
        while (vs[1] == vs[0]) vs[1] = mtrand(0, problem.n_rows - 1);

        arma::mat v =
            arma::trans(safe_normalize(arma::cross(problem.row(vs[0]), problem.row(vs[1]))));

        arma::mat residuals = nproblem * v;
        arma::mat residuals2 = residuals % residuals;

        std::sort(residuals2.begin(), residuals2.end());
        double med = residuals2(residuals2.n_rows / 4, 0);
        if (med < least_med) {
            least_med = med;
            best_sol = v;
        }
    }
    return best_sol;
}

std::pair<double, double> pre_sync(OptData& opt_data, int frame_begin, int frame_end,
                                   double rough_delay, double search_radius, double step) {
    std::vector<std::pair<double, double>> results;
    std::vector<int> frames;
    for (auto& [frame, _] : opt_data.frame_data) {
        if (frame < frame_begin || frame >= frame_end) continue;
        frames.push_back(frame);
    }
    for (double delay = rough_delay - search_radius; delay < rough_delay + search_radius;
         delay += step) {
        std::mutex mtx;
        double cost{};
        std::for_each(std::execution::par, frames.begin(), frames.end(),
                      [frame_begin, frame_end, delay, &opt_data, &cost, &mtx](int frame) {
                          arma::mat P = opt_compute_problem(frame, delay, opt_data);
                          arma::mat M = opt_guess_translational_motion(P, 20);
                          double k = 1 / arma::norm(P * M) * 1e2;
                          arma::mat r = (P * M) * (k / arma::norm(M));
                          arma::mat rho = arma::log1p(r % r);
                          std::unique_lock<std::mutex> lock(mtx);
                          cost += sqrt(arma::accu(arma::sqrt(rho)));
                      });
        results.emplace_back(cost, delay);
    }
    return *std::min_element(results.begin(), results.end());
}

void FrameState::Loss(const arma::mat& gyro_delay, const arma::mat& motion_estimate,
                      arma::mat& loss, arma::mat& jac_gyro_delay, arma::mat& jac_motion_estimate) {
    arma::mat P = opt_compute_problem(frame_, gyro_delay[0], *problem_);
    arma::mat loss_l, loss_r;
    Loss(gyro_delay - kNumericDiffStep, motion_estimate, loss_l);
    Loss(gyro_delay + kNumericDiffStep, motion_estimate, loss_r);

    auto [v1, j1] = std::make_tuple(P * motion_estimate, P);
    auto [v2, j2] = sqr_jac(v1);

    auto [v3, j3] = sqr_jac(motion_estimate);
    auto [v4, j4] = sum_jac(v3);
    auto [v5, j5, _] = div_jac(v4, var_k * var_k);

    auto [v6, j6a, j6b] = div_jac(v2, v5[0]);
    auto [v7, j7] = log1p_jac(v6);
    auto [v8, j8] = sum_jac(v7);

    loss = v8;

    jac_gyro_delay = (loss_r - loss_l) / 2 / kNumericDiffStep;

    jac_motion_estimate = j8 * j7 * (j6a * j2 * j1 + j6b * j5 * j4 * j3);
}

void FrameState::Loss(const arma::mat& gyro_delay, const arma::mat& motion_estimate,
                      arma::mat& loss) {
    arma::mat P = opt_compute_problem(frame_, gyro_delay[0], *problem_);
    arma::mat r = (P * motion_estimate) * (var_k / arma::norm(motion_estimate));
    arma::mat rho = arma::log1p(r % r);
    loss = {arma::accu(rho)};
}

arma::vec3 FrameState::GuessMotion(double gyro_delay) const {
    arma::mat P = opt_compute_problem(frame_, gyro_delay, *problem_);
    return opt_guess_translational_motion(P, 200);
}

double FrameState::GuessK(double gyro_delay) const {
    arma::mat P = opt_compute_problem(frame_, gyro_delay, *problem_);
    return 1 / arma::norm(P * motion_vec) * 1e2;
}

void SyncProblemPrivate::SetGyroQuaternions(const double* data, size_t count, double sample_rate,
                                            double first_timestamp) {
    problem.sample_rate = sample_rate;
    problem.quats_start = first_timestamp;
    problem.quats = ndspline::make(arma::mat(const_cast<double*>(data), 4, count, false, true));
}

void SyncProblemPrivate::SetGyroQuaternions(const uint64_t* timestamps_us, const double* quats,
                                            size_t count) {
    static constexpr uint64_t k_uhz_in_hz = 1000000ULL;
    static constexpr uint64_t k_us_in_sec = 1000000ULL;
    uint64_t actual_sr_uhz =
        k_uhz_in_hz * k_us_in_sec * count / (timestamps_us[count - 1] - timestamps_us[0]);
    int rounded_sr =
        int(round(actual_sr_uhz / 50. / k_uhz_in_hz) * 50 * k_uhz_in_hz);  // round to nearest 50hz

    std::vector<uint64_t> new_timestamps_vec;
    for (int sample = std::ceil(timestamps_us[0] * rounded_sr);
         k_us_in_sec * k_uhz_in_hz * sample / rounded_sr < timestamps_us[count - 1]; sample += 1) {
        new_timestamps_vec.push_back(k_us_in_sec * k_uhz_in_hz * sample / rounded_sr);
    }

    arma::mat new_quats(4, new_timestamps_vec.size());
    for (int i = 0; i < new_timestamps_vec.size(); ++i) {
        auto ts = new_timestamps_vec[i];
        size_t idx = std::lower_bound(timestamps_us, timestamps_us + count, ts) - timestamps_us;
        double t =
            1. * (ts - timestamps_us[idx - 1]) / (timestamps_us[idx] - timestamps_us[idx - 1]);
        new_quats.col(i) =
            quat_slerp(arma::vec4(quats + 4 * (idx - 1)), arma::vec4(quats + 4 * idx), t);
    }
    problem.sample_rate = 1. * rounded_sr / k_uhz_in_hz;
    problem.quats_start = 1. * new_timestamps_vec[0] / k_us_in_sec;
    problem.quats = ndspline::make(new_quats);
}

void SyncProblemPrivate::SetTrackResult(int frame, const double* ts_a, const double* ts_b,
                                        const double* rays_a, const double* rays_b, size_t count) {
    auto& flow = problem.frame_data[frame];
    flow.rays_a = arma::mat(const_cast<double*>(rays_a), 3, count, false, true);
    flow.rays_b = arma::mat(const_cast<double*>(rays_b), 3, count, false, true);
    flow.ts_a = arma::mat(const_cast<double*>(ts_a), 1, count, false, true);
    flow.ts_b = arma::mat(const_cast<double*>(ts_b), 1, count, false, true);
}

void SyncProblemPrivate::SetFps(double fps) { problem.fps = fps; }

double SyncProblemPrivate::PreSync(double initial_delay, int frame_begin, int frame_end,
                                   double search_step, double search_radius) {
    return pre_sync(problem, frame_begin, frame_end, initial_delay, search_radius, search_step)
        .second;
}

double SyncProblemPrivate::Sync(double initial_delay, int frame_begin, int frame_end) {
    arma::mat gyro_delay(1, 1);
    gyro_delay[0] = initial_delay;

    std::vector<std::unique_ptr<FrameState>> costs;
    for (auto& [frame, _] : problem.frame_data) {
        if (frame < frame_begin || frame > frame_end) continue;
        costs.push_back(std::make_unique<FrameState>(frame, &problem));
        costs.back()->motion_vec = costs.back()->GuessMotion(gyro_delay[0]);
        costs.back()->var_k = costs.back()->GuessK(gyro_delay[0]);
    }

    Backtrack delay_optimizer;
    delay_optimizer.SetHyper(2e-4, .1, 1e-3, 10);

    delay_optimizer.SetObjective([&](arma::vec x) {
        std::mutex m;
        arma::mat cost(1, 1), delay_g(1, 1);
        std::for_each(std::execution::par, costs.begin(), costs.end(),
                      [x, &cost, &delay_g, &m](auto& fs) {
                          arma::mat cur_cost, cur_delay_g, tmp;
                          fs->Loss(x, fs->motion_vec, cur_cost, cur_delay_g, tmp);
                          std::unique_lock<std::mutex> lock(m);
                          cost += cur_cost;
                          delay_g += cur_delay_g;
                      });
        return std::make_pair(cost[0], delay_g);
    });

    delay_optimizer.SetObjective([&](arma::vec x) {
        std::mutex m;
        arma::mat cost(1, 1);
        std::for_each(std::execution::par, costs.begin(), costs.end(), [x, &cost, &m](auto& fs) {
            arma::mat cur_cost;
            fs->Loss(x, fs->motion_vec, cur_cost);
            std::unique_lock<std::mutex> lock(m);
            cost += cur_cost;
        });
        return cost[0];
    });

    struct delay_opt_info {
        double step_size;
    };

    constexpr double delay_b{.3};
    arma::mat delay_v(1, 1);
    auto do_opt_motion = [&]() {
        std::for_each(std::execution::par, costs.begin(), costs.end(), [gyro_delay](auto& fs) {
            ens::L_BFGS lbfgs;
            lbfgs.MaxIterations() = 200;
            lbfgs.MinGradientNorm() = 1e-4;

            struct OptimizedFunction {
                OptimizedFunction(FrameState* fs, arma::mat gyro_delay)
                    : fs(fs), gyro_delay(gyro_delay) {}

                double Evauluate(const arma::mat& x) {
                    arma::mat cost;
                    fs->Loss(gyro_delay, x, cost);
                    return cost[0];
                }

                void Gradient(const arma::mat& x, arma::mat& grad) {
                    EvaluateWithGradient(x, grad);
                }

                double EvaluateWithGradient(const arma::mat& x, arma::mat& grad) {
                    arma::mat cost, del_jac;
                    fs->Loss(gyro_delay, x, cost, del_jac, grad);
                    grad = grad.t();
                    return cost[0];
                }

                FrameState* fs;
                arma::mat gyro_delay;
            };

            OptimizedFunction fun{fs.get(), gyro_delay};
            lbfgs.Optimize(fun, fs->motion_vec);
        });
    };

    auto do_opt_delay = [&]() {
        arma::mat step = delay_optimizer.Step(gyro_delay - delay_b * delay_v);

        delay_v = delay_b * delay_v + step;
        gyro_delay += delay_v;

        return delay_opt_info{arma::norm(step)};
    };

    int converge_counter = 0;

    for (int i = 0; i < 400; i++) {
        // Optimize motion
        do_opt_motion();

        // Optimize delay
        auto info = do_opt_delay();

        if (info.step_size < 1e-4) {
            converge_counter++;
        } else {
            converge_counter = 0;
        }

        if (converge_counter > 5) {
            break;
        }

        std::cerr << gyro_delay[0] << " " << info.step_size << std::endl;
    }

    return gyro_delay[0];
}

void SyncProblemPrivate::DebugPreSync(double initial_delay, int frame_begin, int frame_end,
                                      double search_radius, double* delays, double* costs,
                                      int point_count) {
    std::vector<int> frames;
    for (auto& [frame, _] : problem.frame_data) {
        if (frame < frame_begin || frame >= frame_end) continue;
        frames.push_back(frame);
    }
    for (int i = 0; i < point_count; ++i) {
        double delay = initial_delay - search_radius + 2 * search_radius * i / (point_count - 1);
        std::mutex mtx;
        double cost{};
        std::for_each(std::execution::par, frames.begin(), frames.end(),
                      [this, frame_begin, frame_end, delay, &cost, &mtx](int frame) {
                          arma::mat P = opt_compute_problem(frame, delay, problem);
                          arma::mat M = opt_guess_translational_motion(P, 20);
                          double k = 1 / arma::norm(P * M) * 1e2;
                          arma::mat r = (P * M) * (k / arma::norm(M));
                          arma::mat rho = arma::log1p(r % r);
                          std::unique_lock<std::mutex> lock(mtx);
                          cost += sqrt(arma::accu(arma::sqrt(rho)));
                      });
        delays[i] = delay;
        costs[i] = cost;
    }
}

ISyncProblem* CreateSyncProblem() { return new SyncProblemPrivate(); }

ISyncProblem::~ISyncProblem() {}