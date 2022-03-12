#include <cassert>
#include <iostream>
#include <armadillo>
#include <fstream>

#include <spline.hpp>

class spline {
   public:
    spline(const arma::vec& y) { set_points(y); };

    double operator()(double x) const;
    double deriv(double x) const;

   private:
    void set_points(const arma::vec& y);

    arma::vec m_y, m_b, m_c, m_d;
};

void spline::set_points(const arma::vec& y) {
    int n = y.size();
    arma::mat A(n, 3);
    m_c.resize(n);
    for (int i = 1; i < n - 1; i++) {
        A(i, 0) = 1.0 / 3.0;
        A(i, 1) = 2.0 / 3.0 * 2.0;
        A(i, 2) = 1.0 / 3.0;
        m_c[i] = y[i + 1] - 2 * y[i] + y[i - 1];
    }

    A(0, 1) = 2.0;
    A(0, 2) = 0.0;
    m_c[0] = 0.0;

    A(n - 1, 1) = 2.0;
    A(n - 1, 0) = 0.0;
    m_c[n - 1] = 0.0;

    for (int i = 0; i < n - 2; ++i) {
        double k = 1. / A(i, 1) * A(i + 1, 0);
        A.row(i + 1).cols(0, 1) -= A.row(i).cols(1, 2) * k;
        m_c[i + 1] -= m_c[i] * k;
    }

    for (int i = n - 1; i > 1; --i) {
        double k = 1. / A(i, 1) * A(i - 1, 2);
        A.row(i - 1).cols(1, 2) -= A.row(i).cols(0, 1) * k;
        m_c[i - 1] -= m_c[i] * k;
    }

    m_c /= A.col(1);

    m_d.resize(n);
    m_b.resize(n);
    for (int i = 0; i < n - 1; i++) {
        m_d[i] = 1.0 / 3.0 * (m_c[i + 1] - m_c[i]);
        m_b[i] = (y[i + 1] - y[i]) - 1.0 / 3.0 * (2.0 * m_c[i] + m_c[i + 1]);
    }

    m_d[n - 1] = 0.0;
    m_b[n - 1] = 3.0 * m_d[n - 2] + 2.0 * m_c[n - 2] + m_b[n - 2];
    m_y = y;
}

double spline::operator()(double x) const {
    size_t idx = std::max(std::min(std::floor(x), m_b.size() + 0.), 0.);
    size_t n = m_b.size();
    double h = x - idx;
    if (x < idx) return (m_c[0] * h + m_b[0]) * h + m_y[0];
    if (x > n - 1) return (m_c[n - 1] * h + m_b[n - 1]) * h + m_y[n - 1];
    return ((m_d[idx] * h + m_c[idx]) * h + m_b[idx]) * h + m_y[idx];
}

double spline::deriv(double x) const {
    size_t idx = std::max(std::min(std::floor(x), m_b.size() + 0.), 0.);
    size_t n = m_b.size();
    double h = x - idx;
    if (x < 0) return 2.0 * m_c[0] * h + m_b[0];
    if (x > n - 1) return 2.0 * m_c[n - 1] * h + m_b[n - 1];
    return (3.0 * m_d[idx] * h + 2.0 * m_c[idx]) * h + m_b[idx];
}

int main() {
    arma::vec pts(15);
    pts.randn();

    spline s(pts);

    std::vector<double> X, Y;
    for (int i = 0; i < pts.size(); ++i) {
        X.push_back(i);
        Y.push_back(pts[i]);
    }
    tk::spline s2;
    // s2.set_boundary(tk::spline::first_deriv, 0, tk::spline::first_deriv, 0);
    s2.set_points(X, Y);

    std::ofstream o_pts("pts.csv");
    std::ofstream o_spline("spline.csv");
    std::ofstream o_spline2("spline2.csv");

    for (int i = 0; i < pts.size(); ++i) {
        o_pts << i << "," << pts[i] << std::endl;
    }

    for (double x = -1; x <= pts.size(); x += .001) {
        // o_spline << x << "," << s.deriv(x) << std::endl;
        // o_spline2 << x << "," << s2.deriv(1,x) << std::endl;
        o_spline << x << "," << s(x) << std::endl;
        o_spline2 << x << "," << s2(x) << std::endl;
    }

    // for (int i = 0; i < X.size(); ++i) {
    //     std::cout << s.m_c[i] << " " << s2.m_c[i] << std::endl;
    // }

    return 0;
}