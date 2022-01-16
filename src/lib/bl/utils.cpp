#include "utils.hpp"

#include <atomic>

namespace rssync {

class UuidGenImpl : public rssync::IUuidGen {
   public:
    void ContextLoaded(std::weak_ptr<BaseComponent> self) override;

    long Next() override;

   private:
    std::atomic_long g{};
};

void UuidGenImpl::ContextLoaded(std::weak_ptr<BaseComponent> self) {}

long UuidGenImpl::Next() { return g.fetch_and(1); }

void RegisterUuidGen(std::shared_ptr<IContext> ctx, std::string name) {
    RegisterComponent<UuidGenImpl>(ctx, name);
}
}  // namespace rssync