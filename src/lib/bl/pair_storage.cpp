#include "pair_storage.hpp"

#include <mutex>
#include <unordered_map>

namespace rssync {

class PairStorageImpl : public IPairStorage {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override {}

    void Update(int frame, const PairDescription& desc) override {
        std::unique_lock<std::mutex> lock{mtx_};
        data_[frame] = desc;
    }

    bool Get(int frame, PairDescription& desc) override {
        std::unique_lock<std::mutex> lock{mtx_};
        bool found = data_.count(frame);
        if (found) {
            desc = data_[frame];
        }
        return found;
    }

    bool Drop(int frame) override {
        std::unique_lock<std::mutex> lock{mtx_};
        return data_.erase(frame);
    }

    void GetFramesWith(std::vector<int> out, bool points, bool undistorted, bool pose,
                       bool points4d, bool correlations) override {
        std::unique_lock<std::mutex> lock{mtx_};
        for (auto& [k, v] : data_) {
            if (v.has_points < points) continue;
            if (v.has_undistorted < undistorted) continue;
            if (v.has_pose < pose) continue;
            if (v.has_points4d < points4d) continue;
            if (v.has_correlations < correlations) continue;
            out.push_back(k);
        }
    }

   private:
    std::unordered_map<int, PairDescription> data_;
    std::mutex mtx_;
};

void RegisterPairStorage(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<PairStorageImpl>(ctx, name);
}

}  // namespace rssync