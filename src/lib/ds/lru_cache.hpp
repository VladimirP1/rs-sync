#pragma once
#include <cassert>
#include <functional>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

template <class Key, class T, class Hash = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>>
class LruCache {
   public:
    LruCache(size_t capacity) : capacity_(capacity) {
        if (capacity <= 0) {
            throw std::invalid_argument{"capacity should be >= 0"};
        }
        hash_.reserve(capacity);
    }

    void put(const Key& k, const T& v) {
        assert(list_.size() <= capacity_);
        assert(list_.size() == hash_.size());

        if (is_full()) drop_one();
        const auto hash_it = hash_.find(k);
        if (hash_it != hash_.end()) {
            auto old_list_it = hash_it->second;
            auto list_it = list_.emplace(list_.end(), k, v);
            try {
                hash_[k] = list_it;
            } catch (...) {
                list_.erase(list_it);
                throw;
            }
            list_.erase(old_list_it);
        } else {
            auto list_it = list_.emplace(list_.end(), k, v);
            try {
                hash_.emplace(k, list_it);
            } catch (...) {
                list_.erase(list_it);
                throw;
            }
        }
        assert(hash_[k]->first == k);
    }

    std::optional<T> get(const Key& k) {
        assert(list_.size() <= capacity_);
        assert(list_.size() == hash_.size());

        const auto hash_it = hash_.find(k);
        if (hash_it != hash_.end()) {
            list_.splice(list_.end(), list_, hash_it->second);

            assert(hash_[k]->first == k);
            
            return hash_it->second->second;
        } else {
            return std::nullopt;
        }
    }

   private:
    using List = std::list<std::pair<Key, T>>;
    using ListIterator = typename List::iterator;

    size_t capacity_;
    List list_;
    std::unordered_map<Key, ListIterator, Hash, KeyEqual> hash_;

    void drop_one() {
        assert(list_.size() > 0);
        assert(list_.size() == hash_.size());

        const auto it = list_.begin();

        assert(hash_.find(it->first) != hash_.end());
        assert(hash_.find(it->first)->second == it);

        hash_.erase(it->first);
        list_.erase(it);
    }

    bool is_full() { return hash_.size() == capacity_; }
};