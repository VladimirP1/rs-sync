#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <type_traits>

template <typename Base>
class Interpolated : public Base {
    typedef typename Base::group_type G;

   public:
    template <typename I,
              std::enable_if_t<
                  std::is_same<typename std::iterator_traits<I>::value_type,
                               typename G::value_type>::value,
                  bool> = true>
    Interpolated(I begin, I end) : Base(begin, end) {
        data_size_ = std::distance(begin, end);
    }

    typename G::value_type SoftQuery(double l, double r) {
        typename G::value_type ret = group_.unit();
        l = std::min(std::max(l, 0.), static_cast<double>(data_size_));
        r = std::min(std::max(r, 0.), static_cast<double>(data_size_));

        if (l >= r) return group_.unit();

        double disc_l = std::ceil(l);
        double disc_r = std::floor(r);

        if (disc_l < disc_r) {
            size_t disc_rim =
                std::min(std::max(static_cast<ssize_t>(disc_r - 1),
                                  static_cast<ssize_t>(0)),
                         static_cast<ssize_t>(data_size_) - 1);
            size_t disc_lim = std::min(
                std::max(static_cast<ssize_t>(disc_l), static_cast<ssize_t>(0)),
                static_cast<ssize_t>(data_size_) - 1);

            ret = Base::QueryImpl(disc_lim, disc_rim);
        }

        if (((int)disc_l) <= (int)disc_r) {
            ret = group_.add(SubQuery(l, disc_l), ret);
            ret = group_.add(ret, SubQuery(disc_r, r));
        } else {
            ret = SubQuery(l, r);
        }

        return ret;
    }

   private:
    typename G::value_type SubQuery(double l, double r) {
        if (l == r) return group_.unit();
        double base;
        if (std::fabs(l - std::round(l)) > std::fabs(r - std::round(r))) {
            base = l;
        } else {
            base = r;
        }

        double k = std::fabs(r - l);
        ssize_t idx = std::floor(base);

        assert(idx >= 0 && idx < data_size_);
        assert(k >= 0 && k <= 1);

        return group_.mult(Base::QueryImpl(idx, idx), k);
    }

   private:
    G group_;
    size_t data_size_;
};