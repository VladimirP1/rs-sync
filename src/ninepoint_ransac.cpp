#include <Eigen/Eigen>
#include <unsupported/Eigen/src/AutoDiff/AutoDiffScalar.h>

#include <ceres/tiny_solver.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <tuple>
#include <set>
#include <fstream>

struct Points {
    Eigen::ArrayXd x, y, z;
    Eigen::ArrayXd u_x, u_y, u_z;
    Eigen::ArrayXd l1, l2;
    Eigen::ArrayXd betas;
};

struct DiffReprojectionError {
    using AdScalar = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    using AdVector3 = Eigen::Matrix<AdScalar, 3, 1>;
    using AdMatrixX = Eigen::Matrix<AdScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using AdArrayX = Eigen::Array<AdScalar, Eigen::Dynamic, Eigen::Dynamic>;

    typedef double Scalar;
    enum { NUM_RESIDUALS = Eigen::Dynamic, NUM_PARAMETERS = Eigen::Dynamic };

    bool operator()(const double* parameters, double* residuals, double* jacobian) const {
        AdScalar k;
        AdVector3 w, v;
        AdArrayX Z(NumResiduals(), 1);
        //clang-format off
        w << AdScalar{parameters[0], NumParameters(), 0},
            AdScalar{parameters[1], NumParameters(), 1},
            AdScalar{parameters[2], NumParameters(), 2};
        v << AdScalar{parameters[3], NumParameters(), 3},
            AdScalar{parameters[4], NumParameters(), 4},
            AdScalar{parameters[5], NumParameters(), 5};
        k = AdScalar{parameters[6], NumParameters(), 6};
        for (int i = 0; i < NumResiduals(); ++i) {
            Z(i, 0) = AdScalar{parameters[i + 7], NumParameters(), i + 7};
        }

        AdArrayX betas = l1 + (l2 - l1) * k;
        AdArrayX Av(2, p_x.rows()), Bw(2, p_x.rows());
        Av.row(0) = (p_x * v(2) - v(0)).transpose();
        Av.row(1) = (p_y * v(2) - v(1)).transpose();
        Bw.row(0) = (p_x * p_y * w(0) - (1 + p_x * p_x) * w(1) + p_y * w(2)).transpose();
        Bw.row(1) = ((1 + p_y * p_y) * w(0) - p_x * p_y * w(1) - p_x * w(2)).transpose();
        Av.row(0) /= Z.transpose();
        Av.row(1) /= Z.transpose();
        Av += Bw;
        Av.row(0) *= betas.transpose();
        Av.row(1) *= betas.transpose();
        // std::cout << u_x.transpose() << std::endl;
        // std::cout << Av << std::endl;
        // std::cout << u_y.transpose() << std::endl;
        // std::cout << "---------\n";
        Av.row(0) -= u_x.transpose();
        Av.row(1) -= u_y.transpose();

        AdArrayX res = Av.matrix().colwise().norm();
        // std::cout << res << "\n---" << std::endl;

        if (residuals) {
            for (int i = 0; i < p_x.rows(); ++i) {
                residuals[i] = res(0,i).value();
            }
        }

        if (jacobian) {
            Eigen::Map<Eigen::MatrixXd> jac(jacobian, p_x.rows(), p_x.rows() + 7);
            for (int i = 0; i < p_x.rows(); ++i) {
                jac.row(i) = res(0,i).derivatives().transpose();
                jac.row(i).head(7).setZero();
                // std::cout << jac.row(i) << std::endl;
            }
        }

        //clang-format on
        return true;
    }

    DiffReprojectionError(const Points& pts) {
        p_x = pts.x.cast<AdScalar>();
        p_y = pts.y.cast<AdScalar>();
        p_z = pts.z.cast<AdScalar>();

        u_x = pts.u_x.cast<AdScalar>();
        u_y = pts.u_y.cast<AdScalar>();
        u_z = pts.u_z.cast<AdScalar>();

        l1 = pts.l1.cast<AdScalar>();
        l2 = pts.l2.cast<AdScalar>();
    }

    int NumResiduals() const { return p_x.rows(); }

    int NumParameters() const { return p_x.rows() + 7; }

    static void MakeParamBlock(Eigen::Vector3d w, Eigen::Vector3d v, double k, int n,
                               Eigen::Matrix<double, Eigen::Dynamic, 1>& params) {
        params.resize(n + 7, 1);
        params.block(0, 0, 3, 1) = w;
        params.block(3, 0, 3, 1) = v;
        params(6, 0) = k;
        params.block(7, 0, n, 1) = Eigen::ArrayXd::Ones(n, 1) * 1;
    }

   private:
    AdArrayX p_x, p_y, p_z, u_x, u_y, u_z, l1, l2;
};

bool LoadPair(std::string filename, int frame, Eigen::Array2Xd& p, Eigen::Array2Xd& u,
              Eigen::Array2Xd& ts) {
    std::ifstream in(filename);
    int total, tmp;
    bool loaded{};
    // total frame count
    in >> total;

    for (int i = 0; i < total; ++i) {
        // total points in frame
        in >> tmp;
        if (i == frame) {
            std::cout << tmp << std::endl;
            p.resize(2, tmp);
            u.resize(2, tmp);
            ts.resize(2, tmp);
            loaded = true;
        }
        // Load the points
        for (int j = 0; j < tmp; ++j) {
            double x1, y1, x2, y2, t1, t2;
            in >> x1 >> y1 >> x2 >> y2 >> t1 >> t2;
            if (i == frame) {
                p.col(j) << x1, y1;
                u.col(j) << x2 - x1, y2 - y1;
                t2 += 1.;
                ts.col(j) << t1, t2;
            }
        }
    }
    return loaded;
}

void PreparePoints(const Eigen::Array2Xd& p, const Eigen::Array2Xd& u, const Eigen::Array2Xd& ts,
                   Points& points) {
    points.x = p.row(0);
    points.y = p.row(1);
    points.z = Eigen::ArrayXd::Ones(p.cols(), 1);

    points.u_x = u.row(0);
    points.u_y = u.row(1);
    points.u_z = Eigen::ArrayXd::Zero(p.cols(), 1);

    points.l1 = ts.row(1) - ts.row(0);
    points.l2 = ts.row(1) * ts.row(1) - ts.row(0) * ts.row(0);

    // points.l1 = Eigen::ArrayXd::Ones(p.cols(), 1);
    // points.l2 = Eigen::ArrayXd::Ones(p.cols(), 1);
    // points.u_x = Eigen::ArrayXd::Zero(p.cols(), 1);
    // points.u_y = Eigen::ArrayXd::Zero(p.cols(), 1);
    // points.u_z = Eigen::ArrayXd::Zero(p.cols(), 1);
    // std::cout << "\n" << u.row(1) << std::endl << points.u_y.transpose() << std::endl;
}

void CalculateABk(const Points& p, Eigen::MatrixXd& A, Eigen::MatrixXd& B, double& k) {
    Eigen::MatrixXd M(p.x.rows(), 9);
    M.col(0) = -p.x * p.x;
    M.col(1) = -2 * p.x * p.y;
    M.col(2) = -2 * p.x * p.z;
    M.col(3) = -p.y * p.y;
    M.col(4) = -2 * p.y * p.z;
    M.col(5) = -p.z * p.z;
    M.rightCols(3).setZero();

    A = M.array().colwise() * p.l2.array() - M.array().colwise() * p.l1.array();

    B = M.array().colwise() * p.l1.array();
    B.col(6) = p.u_z * p.y - p.u_y * p.z;
    B.col(7) = p.u_x * p.z - p.u_z * p.x;
    B.col(8) = p.u_y * p.x - p.u_x * p.y;

    Eigen::MatrixXd L1 = A.topLeftCorner(6, 6) - A.topRightCorner(6, 3) *
                                                     B.bottomRightCorner(3, 3).inverse() *
                                                     A.bottomLeftCorner(3, 6);
    Eigen::MatrixXd L0 = B.topLeftCorner(6, 6) - B.topRightCorner(6, 3) *
                                                     B.bottomRightCorner(3, 3).inverse() *
                                                     B.bottomLeftCorner(3, 6);

    Eigen::MatrixXd L = L0 * L1.inverse();

    Eigen::EigenSolver<Eigen::MatrixXd> solver(L);
    Eigen::Matrix<std::complex<double>, 6, 1> k_sols = solver.eigenvalues();

    k = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 6; ++i) {
        double cur = k_sols(i, 0).real();
        if (fabs(k_sols(i, 0).imag()) < 1e-7 && fabs(cur) < 1 && fabs(cur) < fabs(k)) {
            k = k_sols(i, 0).real();
        }
    }

    if (!std::isfinite(k)) {
        k = 0;
    }
    // std::cout << k << std::endl;

    // Eigen::MatrixXd betas = -p.l1 * k + p.l1 + p.l2 * k;

    // std::cout << k_sols << std::endl;
    // std::cout << "\n" << p.u_y << std::endl;
    // std::cout << (A * k + B).determinant() << std::endl;
}

void CalculateWV(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double k, Eigen::Vector3d& w,
                 Eigen::Vector3d& vel) {
    Eigen::MatrixXd C = B + k * A;

    Eigen::Vector3d v0;
    Eigen::Matrix3d s;
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> solver(C, Eigen::ComputeFullV);
        Eigen::MatrixXd V = solver.matrixV();
        Eigen::Matrix<double, 9, 1> e = V.rightCols(1);
        v0 = e.tail(3);
        e /= v0.norm();
        s << e(0), e(1), e(2), e(1), e(3), e(4), e(2), e(4), e(5);
    }
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(s);
        Eigen::Vector3d eig_val = solver.eigenvalues();
        Eigen::Matrix3d eig_vec = solver.eigenvectors();
        Eigen::Vector3d sigma;
        sigma << (2 * eig_val(0) + eig_val(1) - eig_val(2)) / 3,
            (2 * eig_val(1) + eig_val(0) + eig_val(2)) / 3,
            (2 * eig_val(2) + eig_val(1) - eig_val(0)) / 3;

        double lambda = sigma(2) - sigma(0);
        double theta = lambda > 1e-6 ? acos(-sigma(1) / lambda) : 0.;

        Eigen::Matrix3d s_1 = Eigen::Vector3d({1, 1, 0}).asDiagonal();
        Eigen::Matrix3d s_lam = s_1 * lambda;
        Eigen::Matrix3d r_v =
            Eigen::AngleAxisd((theta - M_PI) / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d r_u = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()).toRotationMatrix();
        static const Eigen::Matrix3d r_za =
            Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        static const Eigen::Matrix3d r_zb = r_za.transpose();
        Eigen::Matrix3d v = eig_vec * r_v.transpose();
        Eigen::Matrix3d u = -v * r_u;

        auto uv = {std::make_pair(u, v), std::make_pair(v, u)};
        auto zz = {r_za, r_zb};

        Eigen::Vector3d best_vel;
        Eigen::Matrix3d sel_u, sel_v, sel_rz;
        double best_dot = -std::numeric_limits<double>::infinity();
        for (auto& [u_, v_] : uv) {
            for (auto& rz : zz) {
                Eigen::Matrix3d vh = v_ * rz * s_1 * v_.transpose();
                Eigen::Vector3d vel;
                vel << vh(2, 1), vh(0, 2), vh(1, 0);

                if (vel.dot(v0) > best_dot) {
                    best_dot = vel.dot(v0);
                    sel_u = u_;
                    sel_v = v_;
                    sel_rz = rz;
                    best_vel = vel;
                }
            }
        }
        Eigen::Matrix3d wh = sel_u * sel_rz * s_lam * sel_u.transpose();
        w << wh(2, 1), wh(0, 2), wh(1, 0);
        vel = v0;
    }
}

int main() {
    Eigen::Array2Xd p, u, ts;
    LoadPair("000458AA_tracking_data.txt", 38 * 30, p, u, ts);
    // LoadPair("000458AA_tracking_data.txt", 3903, p, u, ts);

    Eigen::Array2Xd np(2, 9), nu(2, 9), nts(2, 9);

    srand(11);
    for (int j = 0; j < 10; ++j) {
        std::set<int> selected;
        while (selected.size() < 9) {
            selected.insert(static_cast<size_t>(rand()) % p.cols());
        }
        int i = 0;
        for (auto s : selected) {
            np.col(i) = p.col(s);
            nu.col(i) = u.col(s);
            nts.col(i) = ts.col(s);
            ++i;
        }

        Points pp;
        PreparePoints(np, nu, nts, pp);

        double k;
        Eigen::MatrixXd A, B;
        CalculateABk(pp, A, B, k);

        Eigen::Vector3d v, w;
        CalculateWV(A, B, k, v, w);

        DiffReprojectionError ef(pp);
        Eigen::Matrix<double, Eigen::Dynamic, 1> params;
        ef.MakeParamBlock(w, v, k, pp.x.rows(), params);
        double out[1024];
        ef(params.data(), out, nullptr);
        std::cout << out[0] << std::endl;

        ceres::TinySolver<DiffReprojectionError> solver;
        solver.options.max_num_iterations = 1000;
        auto summary = solver.Solve(ef, &params);

        std::cout << summary.initial_cost << " -> " << summary.final_cost << " (" << summary.status << ")"<< std::endl;

        std::cout << params.transpose() << std::endl;

        // std::cout << pp.u_y << std::endl;
    }
    return 0;
}