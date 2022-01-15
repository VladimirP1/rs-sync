#include <iostream>
#include <stdexcept>

#include <opencv2/videoio.hpp>

#include <ds/sync_context.hpp>
#include <ds/blocking_queue.hpp>
#include <bl/message_types.hpp>
#include <bl/frame_cache.hpp>

class SyncController {
    using EventQueueT = BlockingMulticastQueue<std::shared_ptr<Message>>;

   public:
    SyncController(std::shared_ptr<EventQueueT> queue) : event_bus_(queue) { Initialize(); }

    void Initialize() { subscription_ = event_bus_->Subscribe(); }

    void ProcessEvent(EventMessage* message) {}

    void Run() {
        std::shared_ptr<Message> message;
        while (subscription_.Dequeue(message)) {
            std::cout << message->ToString() << std::endl;
            if (auto event = dynamic_cast<EventMessage*>(message.get()); event) {
                ProcessEvent(event);
            } else {
            }
        }
    }

   private:
    SyncContext sync_ctx_;

    EventQueueT::Subscription subscription_;
    std::shared_ptr<EventQueueT> event_bus_;
};

int main() {
    using EventQueueT = BlockingMulticastQueue<std::shared_ptr<Message>>;
    std::shared_ptr<EventQueueT> event_bus{EventQueueT::Create()};

    SyncController controller{event_bus};
    std::thread z0, z1;
    {
        std::atomic_bool ok{false};
        std::thread t([&event_bus, &ok]() {
            auto loader = FrameLoader::Create(event_bus, "141101AA.MP4");
            ok = true;
            loader->Run();
        });
        while (!ok)
            ;
        z0 = std::move(t);
    }

    {
        std::atomic_bool ok{false};
        std::thread t([&event_bus, &ok]() {
            auto loader = FrameLoader::Create(event_bus, "141101AA.MP4");
            ok = true;
            loader->Run();
        });
        while (!ok)
            ;
        z1 = std::move(t);
    }

    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 3; ++i) {
            event_bus->Enqueue(std::make_shared<LoadFrameTaskMessage>(i + 20 * j));
        }
    }

    controller.Run();

    return 0;
}