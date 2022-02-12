#pragma once

#include "lru_cache.hpp"

template <typename Base>
class RangeQueryCache : public Base {
   public:
    typedef typename Base::group_type group_type;

    RangeQueryCache() : Base(), cache_(3) {}

    template <typename I,
              std::enable_if_t<std::is_same<typename std::iterator_traits<I>::value_type,
                                            typename group_type::value_type>::value,
                               bool> = true>
    RangeQueryCache(size_t cache_size, I begin, I end) : Base(begin, end), cache_(cache_size) {}

    typename group_type::value_type Query(ptrdiff_t l, ptrdiff_t r) const {
        assert(l >= 0 && l < data_.size());
        assert(r >= 0 && r < data_.size());

        if (auto res = cache_.get({l, r})) {
            return res.value();
        }

        auto res = Base::QueryImpl(l, r);
        cache_.put({l, r}, res);
        return res;
    }

   private:
    struct PairHash {
        std::size_t operator()(std::pair<ptrdiff_t, ptrdiff_t> const& p) const noexcept {
            return p.first ^ p.second;
        }
    };

    mutable LruCache<std::pair<ptrdiff_t, ptrdiff_t>, typename group_type::value_type, PairHash> cache_;
};
