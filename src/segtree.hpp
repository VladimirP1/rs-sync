#include <cassert>
#include <cmath>
#include <iterator>
#include <type_traits>
#include <vector>

template <class T>
struct DefaultGroup {
    typedef T value_type;
    T unit() const { return {}; }

    T add(const T& a, const T& b) const { return a + b; }

    T mult(const T& a, double k) const { return a * k; }
};

template <typename G>
class SegmentTree {
   public:
    template <typename I,
              std::enable_if_t<
                  std::is_same<typename std::iterator_traits<I>::value_type,
                               typename G::value_type>::value,
                  bool> = true>
    SegmentTree(I begin, I end) {
        data_size_ = std::distance(begin, end);
        if (data_size_ == 0) { return; }
        size_t dsize = 1;
        while (dsize < data_size_) dsize <<= 1;
        data_.resize(4 * dsize);
        BuildImpl(begin, end, 1, 0, data_size_ - 1);
    }

    typename G::value_type Query(ssize_t l, ssize_t r) {
        assert(l >= 0 && l < data_size_);
        assert(r >= 0 && r < data_size_);

        return QueryImpl(l, r);
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

            ret = QueryImpl(disc_lim, disc_rim);
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
    template <typename I>
    void BuildImpl(I begin, I end, size_t v, size_t l, size_t r) {
        if (l == r) {
            data_[v] = *(begin + l);
        } else {
            size_t mid = (l + r) / 2;
            BuildImpl(begin, end, 2 * v, l, mid);
            BuildImpl(begin, end, 2 * v + 1, mid + 1, r);
            data_[v] = group_.add(data_[2 * v], data_[2 * v + 1]);
        }
    }

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

        return group_.mult(QueryImpl(idx, idx), k);
    }

    typename G::value_type QueryImpl(size_t l, size_t r) {
        return QueryImpl(1, 0, data_size_ - 1, l, r);
    }

    typename G::value_type QueryImpl(size_t v, size_t cl, size_t cr, size_t ql,
                                     size_t qr) const {
        if (ql > qr) {
            return group_.unit();
        }
        if (cl == ql && cr == qr) {
            return data_[v];
        }
        size_t mid = (cl + cr) / 2;
        return group_.add(
            QueryImpl(2 * v, cl, mid, ql, std::min(qr, mid)),
            QueryImpl(2 * v + 1, mid + 1, cr, std::max(ql, mid + 1), qr));
    }

   private:
    G group_;
    size_t data_size_;
    std::vector<typename G::value_type> data_;
};