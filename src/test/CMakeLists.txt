find_package(Catch2 REQUIRED)
add_executable(self self/self.cpp)
target_link_libraries(self Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(self)