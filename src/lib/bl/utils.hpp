#pragma once
#include "component.hpp"

namespace rssync {

class IUuidGen : public BaseComponent {
   public:
    virtual long Next() = 0;
};

void RegisterUuidGen(std::shared_ptr<IContext> ctx, std::string name);

constexpr const char* kUuidGenName = "UuidGen";
}  // namespace rssync