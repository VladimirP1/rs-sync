#pragma once

template <class T>
struct DefaultGroup {
    typedef T value_type;
    T unit() const { return {}; }

    T add(const T& a, const T& b) const { return a + b; }

    T mult(const T& a, double k) const { return a * k; }

    T inv(const T& a) const { return -a; }
};
