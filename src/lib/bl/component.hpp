#pragma once

#include "context.hpp"

namespace rssync {
class IContext;
class Message;

class BaseComponent {
   public:
    BaseComponent(const BaseComponent&) = delete;
    BaseComponent(BaseComponent&&) = delete;
    BaseComponent& operator=(const BaseComponent&) = delete;
    BaseComponent& operator=(BaseComponent&&) = delete;

    virtual void ContextLoaded(std::weak_ptr<BaseComponent> self) = 0;

    virtual ~BaseComponent() {}

   protected:
    BaseComponent() {}
    std::string name_;
    std::weak_ptr<IContext> ctx_;

    template <class T, class... Args>
    friend std::shared_ptr<T> RegisterComponent(std::shared_ptr<IContext> ctx, std::string name,
                                                Args&&... args);
};

template <class T, class... Args>
std::shared_ptr<T> RegisterComponent(std::shared_ptr<IContext> ctx, std::string name,
                                     Args&&... args) {
    auto ptr = std::make_shared<T>(std::forward<Args>(args)...);
    ctx->RegisterComponent(name, ptr);
    ptr->name_ = name;
    ptr->ctx_ = ctx;
    return ptr;
}

}  // namespace rssync