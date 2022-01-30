#pragma once
#include <cassert>
#include <cmath>
#include <iterator>
#include <type_traits>
#include <vector>

template <typename G>
class PrefixSums {
   public:
    typedef G group_type;

    PrefixSums() {}

    template <typename I,
              std::enable_if_t<std::is_same<typename std::iterator_traits<I>::value_type,
                                            typename G::value_type>::value,
                               bool> = true>
    PrefixSums(I begin, I end) {
        data_ = {begin, end};
        BuildImpl();
    }

    typename G::value_type Query(ssize_t l, ssize_t r) const {
        assert(l >= 0 && l < data_.size());
        assert(r >= 0 && r < data_.size());

        return QueryImpl(l, r);
    }

    size_t Size() const { return data_.size(); }

   protected:
    G group_;

    typename G::value_type QueryImpl(size_t l, size_t r) const {
        return group_.add(group_.inv(l ? data_[l - 1] : group_.unit()), data_[r]);
    }

   private:
    std::vector<typename G::value_type> data_;

    void BuildImpl() {
        for (size_t i = 1; i < data_.size(); ++i) {
            data_[i] = group_.add(data_[i - 1], data_[i]);
        }
    }
};
