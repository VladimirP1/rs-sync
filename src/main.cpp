#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

#include "ndspline.hpp"
#include "quat.hpp"
#include "signal.hpp"
#include "cv.hpp"

#include <telemetry-parser.h>

#include <spline.hpp>

struct OptData {
    double quats_start{};
    int sample_rate{};
    ndspline quats{};

    FramesFlow flows{};
};

void optdata_fill_gyro(OptData& optdata, const char* filename, const char* orient) {
    tp_gyrodata data = tp_load_gyro(filename, orient);
    arma::mat timestamps(data.timestamps, 1, data.samples);
    arma::mat gyro(data.gyro, 3, data.samples);
    tp_free(data);
    optdata.sample_rate = gyro_interpolate(timestamps, gyro);

    arma::mat quats(4, data.samples);
    quats.col(0) = {1, 0, 0, 0};
    for (int i = 1; i < quats.n_cols; ++i) {
        quats.col(i) = arma::normalise(
            quat_prod(quats.col(i - 1), quat_from_aa(gyro.col(i) / optdata.sample_rate)));
    }
    optdata.quats_start = timestamps.front();
    optdata.quats = ndspline::make(quats);
}

void opt_run(const OptData& data) {
    double gyro_delay = -.054;

    int i = 0;
    std::vector<arma::mat> motion(data.flows.size());
    for (const auto& [_, flow] : data.flows) {
        arma::mat ap = flow.rows(0, 2);
        arma::mat bp = flow.rows(3, 5);
        arma::mat at = (flow.row(6) - data.quats_start + gyro_delay) * data.sample_rate;
        arma::mat bt = (flow.row(7) - data.quats_start + gyro_delay) * data.sample_rate;

        arma::mat rots(4, at.n_cols);
        arma::mat brp(3, at.n_cols);
        arma::mat problem(at.n_cols, 3);
        arma::mat dproblem(at.n_cols, 3);
        for (int i = 0; i < at.n_cols; ++i) {
            // Gyro integration with detivative wrt time offset
            arma::vec4 a_conj = quat_conj(data.quats.eval(at[i]));
            arma::vec4 da_conj = quat_conj(data.quats.deriv(at[i]));
            arma::vec4 b = data.quats.eval(bt[i]);
            arma::vec4 db = data.quats.deriv(bt[i]);
            double inv_ab_norm = (1. / arma::norm(a_conj)) * (1. / arma::norm(b));
            arma::vec4 rot = quat_prod(a_conj, b) * inv_ab_norm;
            arma::vec4 drot = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) * inv_ab_norm;
            drot -= drot * arma::dot(rot, drot);

            arma::vec3 br = quat_rotate_point(quat_conj(rot), bp.col(i));
            arma::vec3 t = arma::cross(br, drot.rows(1, 3));
            problem.row(i) = arma::trans(arma::cross(ap.col(i), br));
            dproblem.row(i) = arma::trans(arma::cross(ap.col(i), t));
            brp.col(i) = br;
            rots.col(i) = rot;
        }

        if (1) {
            arma::mat nproblem = problem;
            nproblem.each_row([](arma::mat& m) { m /= arma::norm(m); });

            arma::mat weights(problem.n_rows, 1);
            weights = weights.ones();
            arma::vec3 best_sol;
            double least_med = std::numeric_limits<double>::infinity();
            for (int i = 0; i < 200; ++i) {
                int vs[3];
                vs[0] = vs[1] = rand() % problem.n_rows;
                while (vs[1] == vs[0]) vs[2] = vs[1] = rand() % problem.n_rows;
                while (vs[2] == vs[1] || vs[2] == vs[0]) vs[2] = rand() % problem.n_rows;

                arma::mat v = arma::trans(
                    arma::normalise(arma::cross(problem.row(vs[0]) - problem.row(vs[1]),
                                                problem.row(vs[0]) - problem.row(vs[2]))));

                arma::mat residuals = nproblem * v;
                arma::mat residuals2 = residuals % residuals;

                std::sort(residuals2.begin(), residuals2.end());
                double med = residuals2(residuals2.n_rows / 4, 0);
                if (med < least_med) {
                    least_med = med;
                    best_sol = v;
                }
            }
            arma::mat residuals = problem * best_sol;
            arma::mat residuals2 = residuals % residuals;
            auto k = 1e2;
            weights = 1. / (1 + (residuals % residuals) * k * k);
            arma::mat w_residuals = (problem * best_sol) % weights;
            double cost = arma::dot(w_residuals, w_residuals);
            // std::cout << cost << std::endl;
        }

        for (int i = 0; i < problem.n_rows; ++i) {
            std::cout << problem(i, 0) << " " << problem(i, 1) << " " << problem(i, 2) << " "
                      << dproblem(i, 0) << " " << dproblem(i, 1) << " " << dproblem(i, 2)
                      << std::endl;
        }
        i++;
        exit(0);
    }
}

double cost(const OptData& data, double gyro_delay, double& der_sum) {
    double cost_sum = 0;
    der_sum = 0;
    int i = 0;
    std::vector<arma::mat> motion(data.flows.size());
    for (const auto& [_, flow] : data.flows) {
        arma::mat ap = flow.rows(0, 2);
        arma::mat bp = flow.rows(3, 5);
        arma::mat at = (flow.row(6) - data.quats_start + gyro_delay) * data.sample_rate;
        arma::mat bt = (flow.row(7) - data.quats_start + gyro_delay) * data.sample_rate;

        arma::mat rots(4, at.n_cols);
        arma::mat brp(3, at.n_cols);
        arma::mat problem(at.n_cols, 3);
        arma::mat dproblem(at.n_cols, 3);
        for (int i = 0; i < at.n_cols; ++i) {
            // Gyro integration with detivative wrt time offset
            arma::vec4 a_conj = quat_conj(data.quats.eval(at[i]));
            arma::vec4 da_conj = quat_conj(data.quats.deriv(at[i]));
            arma::vec4 b = data.quats.eval(bt[i]);
            arma::vec4 db = data.quats.deriv(bt[i]);
            double inv_ab_norm = (1. / arma::norm(a_conj)) * (1. / arma::norm(b));
            arma::vec4 rot = quat_prod(a_conj, b) * inv_ab_norm;
            arma::vec4 drot = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) * inv_ab_norm;
            drot -= drot * arma::dot(rot, drot);

            arma::vec3 br = quat_rotate_point(quat_conj(rot), bp.col(i));
            arma::vec3 t = arma::cross(br, drot.rows(1, 3));
            problem.row(i) = arma::trans(arma::cross(ap.col(i), br));
            dproblem.row(i) = arma::trans(arma::cross(ap.col(i), t));
            brp.col(i) = br;
            rots.col(i) = rot;
        }

        if (0) {
            arma::mat nproblem = problem;
            nproblem.each_row([](arma::mat& m) { m /= arma::norm(m); });

            arma::vec3 best_sol;
            double least_med = std::numeric_limits<double>::infinity();
            for (int i = 0; i < 200; ++i) {
                int vs[3];
                vs[0] = vs[1] = rand() % problem.n_rows;
                while (vs[1] == vs[0]) vs[2] = vs[1] = rand() % problem.n_rows;
                while (vs[2] == vs[1] || vs[2] == vs[0]) vs[2] = rand() % problem.n_rows;

                arma::mat v = arma::trans(
                    arma::normalise(arma::cross(nproblem.row(vs[0]) - nproblem.row(vs[1]),
                                                nproblem.row(vs[0]) - nproblem.row(vs[2]))));

                arma::mat residuals = nproblem * v;
                arma::mat residuals2 = residuals % residuals;

                std::sort(residuals2.begin(), residuals2.end());
                double med = residuals2(residuals2.n_rows / 4, 0);
                if (med < least_med) {
                    least_med = med;
                    best_sol = v;
                }
            }

            
            const double k = 1e3;
            arma::mat residuals = problem * best_sol;
            // std::cout << arma::trans(residuals) << std::endl;
            arma::mat weights = arma::sqrt(1. / (1. + (residuals % residuals) * k * k));
            // arma::mat weights = k / arma::abs(residuals);
            // weights.transform([](double& x){ return x > 1 ? 1 : x; });
            for (int i = 0; i < 20; ++i) {
                arma::mat U,V;
                arma::vec S;
                arma::svd(U,S,V,problem.each_col() % weights);
                residuals = (problem * V.tail_cols(1));
                weights = arma::sqrt(1. / (1. + (residuals % residuals) * k * k));
                // arma::mat weights = k / arma::abs(residuals);
                // weights.transform([](double& x){ return x > 1 ? 1 : x; });
            }

            problem = problem.each_col() % weights;
            dproblem = dproblem.each_col() % weights;

            // std::cerr << S.t() << std::endl;

            // arma::mat residuals = problem * best_sol;
            // arma::mat residuals2 = residuals % residuals;
            // auto k = 1e2;
            // weights = 1. / (1 + (residuals % residuals) * k * k);
            // arma::mat w_residuals = (problem * best_sol) % weights;
            // double cost = arma::dot(w_residuals, w_residuals);
            // cost_sum += cost;
            // std::cout << cost << std::endl;
        }

        arma::mat U, V;
        arma::vec S;
        arma::svd(U, S, V, problem);


        cost_sum += S.tail(1).eval()[0] * S.tail(1).eval()[0];
        der_sum += (arma::accu((problem * V.tail_cols(1)) % (problem * V.tail_cols(1))));
        // der_sum += (2*arma::accu((dproblem * V.tail_cols(1))));

        // for (int i = 0; i < problem.n_rows; ++i) {
        //     std::cout << problem(i, 0) << " " << problem(i, 1) << " " << problem(i, 2) << " "
        //               << dproblem(i, 0) << " " << dproblem(i, 1) << " " << dproblem(i, 2)
        //               << std::endl;
        // }
        i++;
    }
    return cost_sum;
}

int main() {
    std::cout << std::fixed << std::setprecision(16);

    OptData opt_data;
    // YXZ zYX
    optdata_fill_gyro(opt_data, "GX011338.MP4", "zYX");

    Lens lens = lens_load("lens.txt", "hero6_27k_43");
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 300, 330);
    track_frames(opt_data.flows, lens, "GX011338.MP4", 400, 450);
    // track_frames(opt_data.flows, lens, "GX011338.MP4", 1300, 1330);

    // opt_run(opt_data);

    for (double ofs = -.065; ofs < -.035; ofs += 1e-4) {
        double der;
        std::cout << ofs << "," << cost(opt_data, ofs, der) << "," << der << std::endl;
    }
    // arma::vec4 a_conj = quat_conj(nsp.eval(t_left));
    // arma::vec4 b = nsp.eval(t_right);
    // arma::vec4 da_conj = quat_conj(nsp.deriv(t_left));
    // arma::vec4 db = nsp.deriv(t_right);
    // i2 = quat_prod(a_conj, b) / arma::norm(a_conj) / arma::norm(b);
    // di2 = (quat_prod(da_conj, b) + quat_prod(a_conj, db)) / arma::norm(a_conj) /
    //       arma::norm(b);
    // di2 -= i2 * arma::dot(i2, di2);
}
