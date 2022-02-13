#include <iostream>

#include <Eigen/Eigen>

using std::cout;

template <class M>
M softargmax(const M& a, double k = 1) {
    return ((k * a).exp() / (k * a).exp().sum());
}

template <class M>
auto softmax(const M& a, double k = 1) {
    return (a * softargmax(a, k)).sum();
}

template <class M>
M softabs(const M& a, double k = 1) {
    return (a * (k * a).exp() - a * (k * -a).exp()) / ((k * a).exp() + (k * -a).exp());
}

template <class M>
M softargmedian(const M& a, double k = 1) {
    auto rep = a.replicate(1, a.rows()).eval();
    return softargmax((-softabs((rep.transpose() - rep).eval(), k).rowwise().sum()).eval(), k);
}

template <class M>
auto softmedian(const M& a, double k = 1) {
    return (a * softargmedian(a, k)).sum();
}

int main() {
    Eigen::Array<double, 8, 1> a;
    a << 2, 5, 8, 5, 5, 9, 1, 2;

    a/=100;

    std::cout << a.transpose() << std::endl;

    // Softmax

    double k = 1;
    std::cout << softmedian(a) << std::endl;

    std::sort(a.data(), a.data() + 8);
    std::cout << a.transpose() << std::endl;

    return 0;
}