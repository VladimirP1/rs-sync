#pragma once

#include <memory>

namespace rssync {
class BaseComponent;
class IContext {
   public:
    static std::shared_ptr<IContext> CreateContext();

    virtual void RegisterComponent(std::string name, std::shared_ptr<BaseComponent>) = 0;
    virtual void DeregisterComponent(std::string name) = 0;
    virtual std::shared_ptr<BaseComponent> GetComponent(std::string name) = 0;
    virtual void ContextLoaded() = 0;

    template <class T>
    std::shared_ptr<T> GetComponent(std::string name) {
        return std::dynamic_pointer_cast<T>(GetComponent(name));
    }

   private:
};
}  // namespace rssync