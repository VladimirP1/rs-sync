#include <iostream>
#include <stdexcept>

#include <opencv2/videoio.hpp>

#include <ds/sync_context.hpp>
#include <ds/blocking_queue.hpp>

int main() {
    // auto ctx = std::make_shared<SyncContext>();
    auto mqueue = BlockingMulticastQueue<int>::Create();
    std::vector<std::thread> threads;
    for (int i = 0; i < 100; ++i) {
        std::thread t([&]() {
            int x;
            auto s = mqueue->Subscribe();
            while (s.Dequeue(x)) {
                std::cout << x << std::endl;
            }
        });
        threads.push_back(std::move(t));
    }
    for (int i = 0; i < 10; ++i) {
        mqueue->Enqueue(i);
    }
    mqueue->Terminate();
    
    for (auto& t: threads) {
        t.join();
    }

    return 0;
}