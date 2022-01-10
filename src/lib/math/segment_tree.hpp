#pragma once
#include <cassert>
#include <cmath>
#include <iterator>
#include <type_traits>
#include <vector>

template <typename G>
class SegmentTree {
   public:
    typedef G group_type;

    template <typename I,
              std::enable_if_t<
                  std::is_same<typename std::iterator_traits<I>::value_type,
                               typename G::value_type>::value,
                  bool> = true>
    SegmentTree(I begin, I end) {
        data_size_ = std::distance(begin, end);
        if (data_size_ == 0) {
            return;
        }
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

   protected:
    G group_;

    typename G::value_type QueryImpl(size_t l, size_t r) {
        return QueryImpl(1, 0, data_size_ - 1, l, r);
    }

   private:
    size_t data_size_;
    std::vector<typename G::value_type> data_;

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
};