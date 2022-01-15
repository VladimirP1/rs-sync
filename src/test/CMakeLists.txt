find_package(Catch2 REQUIRED)

add_executable(test_self self/self.cpp)
target_link_libraries(test_self Catch2::Catch2)

add_executable(test_range_interpolated math/range_interpolated.cpp)
target_link_libraries(test_range_interpolated libmath Catch2::Catch2)

add_executable(test_sync_context ds/sync_context.cpp)
target_link_libraries(test_sync_context libmath libds libvision Catch2::Catch2)

add_executable(test_blocking_queue ds/blocking_queue.cpp)
target_link_libraries(test_blocking_queue libds Catch2::Catch2)

add_executable(test_lru_cache ds/lru_cache.cpp)
target_link_libraries(test_lru_cache libds Catch2::Catch2)

include(CTest)
include(Catch)

catch_discover_tests(test_self)
catch_discover_tests(test_range_interpolated)
catch_discover_tests(test_sync_context)
catch_discover_tests(test_blocking_queue)
catch_discover_tests(test_lru_cache)