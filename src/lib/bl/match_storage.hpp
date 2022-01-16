#pragma once

#include "component.hpp"

namespace rssync {
void RegisterMatchStorage(std::shared_ptr<IContext> ctx, std::string name);

class IOpticalFlow : public rssync::BaseComponent {
   public:
    
};

constexpr const char* kMatchStorageName = "MatchStorage";

}  // namespace rssync