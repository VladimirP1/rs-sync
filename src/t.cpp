#include <iostream>
#include <stdexcept>
#include <thread>

#include <bl/frame_loader.hpp>
#include <bl/keypoint_detector.hpp>
namespace rssync {
struct TestComponent : public rssync::BaseComponent {
    void ContextLoaded(std::weak_ptr<BaseComponent> self, std::weak_ptr<IContext> ctx) override {
        t = std::thread(&TestComponent::Run, this);
    }

    void Run() {
        std::shared_ptr<rssync::Message> msg;
        while (ctx_.lock()->GetComponent("testcomponent0")->Inbox().Dequeue(msg)) {
            std::cout << msg->ToString() << std::endl;
        }
    }

    ~TestComponent(){
        Inbox().Terminate();
        t.join();
    }

   private:
    std::thread t;
};
}  // namespace rssync

int main() {
    auto ctx = rssync::IContext::CreateContext();

    rssync::RegisterFrameLoader(ctx, rssync::kFrameLoaderName, 16, "141101AA.MP4");

    rssync::RegisterKeypointDetector(ctx, rssync::kKeypointDetectorName, 16);

    rssync::RegisterComponent<rssync::TestComponent>(ctx, "testcomponent0");

    ctx->ContextLoaded();

    for (int i = 0; i < 800; ++i) {
        // ctx->GetComponent(rssync::kFrameLoaderName)
        //     ->Inbox()
        //     .Enqueue(rssync::MakeMessage<rssync::LoadFrameTaskMessage>("testcomponent0", i));

        ctx->GetComponent(rssync::kKeypointDetectorName)
            ->Inbox()
            .Enqueue(rssync::MakeMessage<rssync::DetectKeypointsTaskMessage>("testcomponent0", i));
    }

    std::cout << "main done" << std::endl;

    return 0;
}