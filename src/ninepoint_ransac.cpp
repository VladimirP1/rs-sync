#include <Eigen/Eigen>
#include <unsupported/Eigen/src/AutoDiff/AutoDiffScalar.h>

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

void CalculateWV(const Points& p, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double k,
                 Eigen::Vector3d& w, Eigen::Vector3d& vel) {
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

        Eigen::Vector3d best_vel, best_w;
        Eigen::Matrix3d sel_u, sel_v, sel_rz;
        int best_pz = 0;
        for (auto& [u_, v_] : uv) {
            for (auto& rz : zz) {
                Eigen::Matrix3d wh = u_ * rz * s_lam * u_.transpose();
                w << wh(2, 1), wh(0, 2), wh(1, 0);
        std::cout << w.transpose() << std::endl;

                Eigen::Matrix3d vh = v_ * rz * s_1 * v_.transpose();
                vel << vh(2, 1), vh(0, 2), vh(1, 0);

                int pz = 0;
                for (int i = 0; i < A.rows(); ++i) {
                    auto x = p.x(i, 0), y = p.y(i, 0);
                    Eigen::Matrix<double, 2, 3> mA, mB;
                    mA << -1, 0, x, 0, -1, y;
                    mB << x * y, -(1 + x * x), y, (1 + y * y), -x * y, -x;
                    auto beta = p.l1(i, 0) * (1 - k) + p.l2(i, 0) * k;
                    Eigen::Matrix<double, 2, 1> u;
                    u << p.u_x(i, 0), p.u_y(i, 0);

                    Eigen::MatrixXd sgn = (mA * vel).transpose() * (u - mB * w);
                    // std::cout << sgn.rows() << "x" << sgn.cols();
                    // std::cout << (mB * w).transpose() << std::endl; 
                    // std::cout << u.transpose() << std::endl;
                    // Eigen::MatrixXd rhs = u - beta * mB * w;
                    // Eigen::MatrixXd lhs = +beta * mA * v;
                    // Eigen::MatrixXd sol =
                    //     lhs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
                    if (sgn(0, 0) >= 0) ++pz;
                }
                std::cout << "pz=" << pz << std::endl;

                if (pz > best_pz) {
                    best_pz = pz;
                    sel_u = u_;
                    sel_v = v_;
                    sel_rz = rz;
                    best_vel = vel;
                    best_w = w;
                }
            }
        }
        // Eigen::Matrix3d vh = sel_v * sel_rz * s_1 * sel_v.transpose();
        // vel << vh(2, 1), vh(0, 2), vh(1, 0);

        Eigen::Matrix3d wh = sel_u * sel_rz * s_lam * sel_u.transpose();
        // std::cout << wh << std::endl;
        w << wh(2, 1), wh(0, 2), wh(1, 0);
        vel = v0;
    }
}

int main() {
    Eigen::Array2Xd p, u, ts;

    p.resize(2,10);
    u.resize(2,10);
    ts.resize(2,10);
    for (int i = 0; i < 10; ++i) {
        Eigen::Vector3d point = Eigen::Vector3d::Random();
        point(2) += 10;
    // std::cout<< point << std::endl;
        Eigen::AngleAxis<double> rot(.1, Eigen::Vector3d{0, 0, 1});
        Eigen::Vector3d point2 = rot * point + Eigen::Vector3d{.1,0,0};

        point /= point(2,0);
        point2 /= point2(2,0);
        p.col(i) << point(0,0), point(1,0);
        u.col(i) << point2(0,0) - point(0,0), point2(1,0) - point(1,0);
        ts.col(i) << 0,1;
    }

    // std::cout<< u << std::endl;
    // LoadPair("000458AA_tracking_data.txt", 38 * 30, p, u, ts);
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

        std::cout << k << std::endl;

        k = 0;

        Eigen::Vector3d w, v;
        CalculateWV(pp, A, B, k, w, v);

        // k = 0;
        /*
        Eigen::Vector3d w, v;
        CalculateWV(pp, A, B, k, w, v);

        v = Eigen::Vector3d::Zero();

        std::cout << w.norm() << " " << v.transpose() << std::endl;
        for (int i = 0; i < 9; ++i) {
            auto x = pp.x(i, 0);
            auto y = pp.y(i, 0);
            auto beta = pp.l1(i, 0) * (1 - k) + pp.l2(i, 0) * k;
            Eigen::Matrix<double, 2, 3> mA, mB;
            mA << -1, 0, x, 0, -1, y;
            mB << x * y, -(1 + x * x), y, (1 + y * y), -x * y, -x;
            Eigen::Matrix<double, 2, 1> u;
            u << pp.u_x(i, 0), pp.u_y(i, 0);
            // std::cout << (mB * w).transpose() << std::endl;
            // std::cout << u.transpose() << std::endl;
            Eigen::MatrixXd rhs = u - beta * mB * w;
            Eigen::MatrixXd lhs = + beta * mA * v;
            Eigen::MatrixXd sol = lhs.jacobiSvd(Eigen::ComputeThinU |
            Eigen::ComputeThinV).solve(rhs);

            std::cout << "\n-------\n";
            std::cout << sol(0,0) << " " << (lhs * sol(0,0) - rhs).norm() * 1700 << std::endl;
        }*/
        // std::cout << pp.u_y << std::endl;
    }
    return 0;
}