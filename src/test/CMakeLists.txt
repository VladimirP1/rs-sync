find_package(Catch2 REQUIRED)

add_executable(self_test self/self.cpp)
target_link_libraries(self_test Catch2::Catch2)

add_executable(range_interpolated math/range_interpolated.cpp)
target_link_libraries(range_interpolated libmath Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(self_test)
catch_discover_tests(range_interpolated)