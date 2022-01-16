#pragma once

#include <ds/blocking_queue.hpp>

#include "context.hpp"
#include "message_types.hpp"

namespace rssync {
class IContext;
class Message;

class BaseComponent {
   public:
    BaseComponent(const BaseComponent&) = delete;
    BaseComponent(BaseComponent&&) = delete;
    BaseComponent& operator=(const BaseComponent&) = delete;
    BaseComponent& operator=(BaseComponent&&) = delete;

    using InboxT = BlockingQueue<std::shared_ptr<Message>>;

    virtual InboxT& Inbox() final { return inbox_; };

    virtual void ContextLoaded(std::weak_ptr<BaseComponent> self, std::weak_ptr<IContext> ctx) = 0;

    virtual ~BaseComponent() { inbox_.Terminate(); }

   protected:
    BaseComponent() {}
    InboxT inbox_;
    std::string name_;
    std::weak_ptr<IContext> ctx_;

    template <class T, class... Args>
    friend std::shared_ptr<T> RegisterComponentLimited(std::shared_ptr<IContext> ctx,
                                                       std::string name, size_t max_size,
                                                       Args&&... args);
};

template <class T, class... Args>
std::shared_ptr<T> RegisterComponentLimited(std::shared_ptr<IContext> ctx, std::string name,
                                            size_t max_size, Args&&... args) {
    auto ptr = std::make_shared<T>(std::forward<Args>(args)...);
    ctx->RegisterComponent(name, ptr);
    ptr->name_ = name;
    ptr->ctx_ = ctx;
    ptr->inbox_.SetMaxSize(max_size);
    return ptr;
}

template <class T, class... Args>
std::shared_ptr<T> RegisterComponent(std::shared_ptr<IContext> ctx, std::string name,
                                     Args&&... args) {
    return RegisterComponentLimited<T>(ctx, name, 0, std::forward<Args>(args)...);
}

}  // namespace rssync