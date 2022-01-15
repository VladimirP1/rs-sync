#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <optional>
#include <string>

#include <ds/lru_cache.hpp>

SCENARIO("LruCache put and get methods work correctly") {
    GIVEN("An LruCache of size 3") {
        LruCache<int, std::string> lru(3);

        REQUIRE_FALSE(lru.get(0).has_value());

        WHEN("Cache gets exhausted by one element") {
            lru.put(1, "1");
            lru.put(2, "2");
            lru.put(3, "3");
            lru.put(4, "4");

            THEN("Only the actual lru key gets removed") {
                REQUIRE(lru.get(1) == std::nullopt);
                REQUIRE(lru.get(2) == "2");
            }
        }

        WHEN("Get operation is mixed with puts") {
            lru.put(1, "1");
            lru.put(2, "2");
            lru.put(3, "3");
            lru.get(1);
            lru.put(4, "4");
            THEN("Only the actual lru key gets removed") {
                REQUIRE(lru.get(2) == std::nullopt);
                REQUIRE(lru.get(1) == "1");
            }
        }

        WHEN("After put operation overwrites older values") {
            lru.put(1, "1");
            lru.put(2, "2");
            lru.put(3, "3");
            lru.put(3, "3a");
            lru.put(1, "1a");
            lru.put(4, "4");

            THEN("Put operation actually works") {
                REQUIRE(lru.get(1) == "1a");
                REQUIRE(lru.get(3) == "3a");
            }

            THEN("Only the actual lru key gets removed") {
                REQUIRE(lru.get(2) == std::nullopt);
                REQUIRE(lru.get(3) == "3a");
            }
        }
    }
}

SCENARIO("LruCache stress-test") {
    GIVEN("An LruCache of size 128") {
        LruCache<int, std::string> lru(128);

        WHEN("1024 puts are made") {
            for (int i = 0; i < 1024; ++i) {
                lru.put(i, std::to_string(i));
            }

            THEN("Only last 128 items are left") {
                int i;
                for (i = 0; i < 1024 - 128; ++i) {
                    REQUIRE(lru.get(i) == std::nullopt);
                }
                for (; i < 1024; ++i) {
                    REQUIRE(lru.get(i) == std::to_string(i));
                }
            }
        }

        WHEN("1024 ops are made") {
            for (int i = 0; i < 1024; ++i) {
                lru.put(i, std::to_string(i));
                lru.get(i % 64);
            }
            THEN("The correct items get removed") {
                int i;
                for (i = 0; i < 64; ++i) {
                    REQUIRE(lru.get(i) == std::to_string(i));
                }
                for (i = 64; i < 1024 - 64; ++i) {
                    REQUIRE(lru.get(i) == std::nullopt);
                }
                for (i = 1024 - 64; i < 1024; ++i) {
                    REQUIRE(lru.get(i) == std::to_string(i));
                }
            }
        }
    }
}

SCENARIO("LruCache of size zero throws") {
    REQUIRE_THROWS(LruCache<int, std::string>(0));
}