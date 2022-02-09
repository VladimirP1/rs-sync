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

    SegmentTree() {}

    template <typename I,
              std::enable_if_t<std::is_same<typename std::iterator_traits<I>::value_type,
                                            typename G::value_type>::value,
                               bool> = true>
    SegmentTree(I begin, I end) {
        size_t data_size = std::distance(begin, end);
        if (data_size == 0) {
            return;
        }
        data_.resize(2 * data_size);
        BuildImpl(begin, end);
    }

    typename G::value_type Query(ptrdiff_t l, ptrdiff_t r) const {
        assert(l >= 0 && l < data_.size() / 2);
        assert(r >= 0 && r < data_.size() / 2);

        return QueryImpl(l, r);
    }

    size_t Size() const { return data_.size() / 2; }

   protected:
    G group_;

    typename G::value_type QueryImpl(size_t l, size_t r) const {
        l += data_.size() / 2;
        r += data_.size() / 2;

        if (l == r) {
            return data_[l];
        }

        ++r;
        typename G::value_type sum = group_.unit();
        while (l < r) {
            if (l & 1) sum = group_.add(data_[l++], sum);
            if (r & 1) sum = group_.add(sum, data_[--r]);
            l /= 2;
            r /= 2;
        }
        return sum;
    }

   private:
    std::vector<typename G::value_type> data_;

    template <typename I>
    void BuildImpl(I begin, I end) {
        const size_t data_size = data_.size() / 2;
        std::copy(begin, end, data_.begin() + data_size);
        for (size_t i = data_size - 1; i >= 1; i--) {
            data_[i] = group_.add(data_[2 * i], data_[2 * i + 1]);
        }
    }
};