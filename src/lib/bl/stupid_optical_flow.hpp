#pragma once

#include "component.hpp"

namespace rssync {
void RegisterOpticalFlowStupid(std::shared_ptr<IContext> ctx, std::string name);

// class IOpticalFlow : public rssync::BaseComponent {
//    public:
//     virtual bool CalcOptflow(int frame_number) = 0;
// };

}  // namespace rssync