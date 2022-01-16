#include "context.hpp"
#include "component.hpp"

#include <unordered_map>

namespace rssync {
class ContextImpl : public IContext {
   public:
    void RegisterComponent(std::string name, std::shared_ptr<BaseComponent>) override;
    void DeregisterComponent(std::string name) override;
    std::shared_ptr<BaseComponent> GetComponent(std::string name) override;
    void ContextLoaded() override;

   private:
    std::unordered_map<std::string, std::shared_ptr<BaseComponent>> id_to_ptr_{};
    std::weak_ptr<IContext> self_{};

    void SetSelf(std::weak_ptr<IContext> ctx) { self_ = ctx; }
    friend std::shared_ptr<IContext> IContext::CreateContext();
};

void ContextImpl::RegisterComponent(std::string name, std::shared_ptr<BaseComponent> component) {
    id_to_ptr_.insert({name, component});
}

void ContextImpl::DeregisterComponent(std::string name) { id_to_ptr_.erase(name); }

std::shared_ptr<BaseComponent> ContextImpl::GetComponent(std::string name) {
    return id_to_ptr_.at(name);
}

void ContextImpl::ContextLoaded() {
    auto self = self_.lock();
    for (auto& [k, v] : id_to_ptr_) {
        v->ContextLoaded(v, self);
    }
}

std::shared_ptr<IContext> IContext::CreateContext() {
    auto ptr = std::make_shared<ContextImpl>();
    ptr->SetSelf(ptr);
    return ptr;
}

}  // namespace rssync