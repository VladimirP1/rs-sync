#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <thread>

#include <ds/blocking_queue.hpp>

TEST_CASE("BlockingMulticastQueue") {
    SECTION("No deadlock on dequeue from empty queue + terminate") {
        int x;
        auto mqueue = BlockingMulticastQueue<int>::Create();

        std::thread t([&]() {
            auto s = mqueue->Subscribe();
            REQUIRE_FALSE(s.Dequeue(x));
        });
        mqueue->Terminate();
        t.join();
    }

    SECTION("Stress test with 100 subscribers") {
        int x;
        auto mqueue = BlockingMulticastQueue<int>::Create();
        std::vector<std::thread> threads;
        std::atomic_int counter{0}, sub_counter{0};
        for (int i = 0; i < 100; ++i) {
            std::thread t([&]() {
                int x;
                auto s = mqueue->Subscribe();
                sub_counter.fetch_add(1);
                while (s.Dequeue(x)) {
                    counter.fetch_add(1);
                }
            });
            threads.push_back(std::move(t));
        }
        while(sub_counter < threads.size());

        for (int i = 0; i < 10; ++i) {
            mqueue->Enqueue(i);
            counter.fetch_add(-1 * threads.size());
        }

        mqueue->Seal();
        for (auto& t : threads) {
            t.join();
        }

        REQUIRE(counter == 0);
    }
}