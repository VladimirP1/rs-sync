#include <iostream>

#include <Eigen/Eigen>

#include <ceres/ceres.h>
#include <ceres/tiny_solver.h>
#include <unsupported/Eigen/src/AutoDiff/AutoDiffScalar.h>

using std::cout;

struct HomogenousLeastSquaresCost {
    typedef double Scalar;
    using ad = Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1>>;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,
        NUM_PARAMETERS = Eigen::Dynamic,
    };

    HomogenousLeastSquaresCost(const Eigen::MatrixXd& problem) {
        problem_.resize(problem.rows(), problem.cols());
        for (int i = 0; i < problem.rows() * problem.cols(); ++i) {
            problem_.data()[i].value() = problem.data()[i];
            problem_.data()[i].derivatives().resize(problem.cols(), 1);
            problem_.data()[i].derivatives().setZero();
        }
    }

    bool operator()(const double* parameters_, double* residuals_, double* jacobian_) const {
        Eigen::Matrix<ad, -1, -1> parameters(problem_.cols(), 1);
        Eigen::Map<Eigen::Matrix<double, -1, -1>> residuals(residuals_, problem_.rows(), 1);

        for (int i = 0; i < problem_.cols(); ++i) {
            parameters(i, 0).value() = parameters_[i];
            parameters(i, 0).derivatives().resize(problem_.cols(), 1);
            parameters(i, 0).derivatives().setZero();
            parameters(i, 0).derivatives()(i, 0) = 1;
        }

        parameters /= parameters.norm();

        auto result = (problem_ * parameters).eval();
        result = result.array() / (ad(1.) + result.cwiseAbs().array());

        for (int i = 0; i < problem_.rows(); ++i) {
            residuals(i, 0) = result(i, 0).value();
        }

        if (jacobian_) {
            Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> jacobian(
                jacobian_, problem_.rows(), problem_.cols());
            for (int i = 0; i < problem_.rows(); ++i) {
                jacobian.row(i) = result(i, 0).derivatives();
            }
        }

        return true;
    }

    int NumResiduals() const { return problem_.rows(); }

    int NumParameters() const { return problem_.cols(); }

   private:
    Eigen::Matrix<ad, -1, -1> problem_;
};

int main() {
    Eigen::MatrixXd P(200, 3);
    Eigen::VectorXd t(3);
    t.setRandom();
    t.normalize();
    // P.setRandom();
    for (int i = 0; i < P.rows(); ++i) {
        P.row(i) << 1,1,0;
        P.row(i) += Eigen::Vector3d::Random() * .1;
    }
    P.row(0) << 1000, 0, 0;
    P.row(1) << 0, 1000, 0;
    P.row(2) << 0, 0, 1000;
    P.row(3) << 0, 0, 10000;
    P.row(4) << 10000, -10000, 0;
    HomogenousLeastSquaresCost f(P);
    ceres::TinySolver<HomogenousLeastSquaresCost> solver;
    solver.options.max_num_iterations = 500;

    solver.Solve(f, &t);
    std::cout << solver.summary.iterations << " " << solver.summary.initial_cost << " "
              << solver.summary.final_cost << " " << solver.summary.status << " | "
              << t.normalized().transpose() << std::endl;

    Eigen::Matrix<double,-1,1> weights(P.rows(), 1);
    weights.setOnes();
    for (int i = 0; i < 10; ++i) {
        auto svd = (P.array().colwise() * weights.array()).matrix().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto V = svd.matrixV().eval();
        t = V.col(V.cols() - 1).eval();
        auto residuals = (P * t).array().eval();
        weights = 1 / (1. + residuals.cwiseAbs());
        std::cout << t.transpose() << std::endl;
    }

    solver.Solve(f, &t);
    std::cout << solver.summary.iterations << " " << solver.summary.initial_cost << " "
              << solver.summary.final_cost << " " << solver.summary.status << " | "
              << t.normalized().transpose() << std::endl;

    // for (int i = 0; i < 1000; ++i) {
    //     t.setRandom();
    //     t.normalize();
    //     solver.Solve(f, &t);
    //     std::cout << solver.summary.iterations << " " << solver.summary.initial_cost << " "
    //               << solver.summary.final_cost << " " << solver.summary.status << " | "
    //               << t.normalized().transpose() << std::endl;
    // }

    return 0;
}