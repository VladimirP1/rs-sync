#include <iostream>
#include <stdexcept>

#include <opencv2/videoio.hpp>

#include <ds/sync_context.hpp>
#include <bl/async_frame_loader.hpp>

int main() {
    auto ctx = std::make_shared<SyncContext>();
    auto loader = AsyncFrameLoader::CreateFrameLoader("141101AA.MP4");
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 2; ++i) {
        futures.emplace_back(loader->LoadFramesIntoContext(i*20, i*20 + 10, ctx));
    }

    for (auto&f : futures) {
        f.wait();
        std::cout << "." << std::endl;
    }
    

    for (int i = 0; i < 200; ++i) {
        ctx->InFrameContext<void>(i, [i](const FrameContext& ctx) {
            if (ctx.CachedFrame().cols > 0) {
                std::cout << "q " << i << std::endl;
            }
        });
    }

    return 0;
}