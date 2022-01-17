project(rs_sync)

# opencv
find_package(OpenCV REQUIRED)

# nlohmann
find_package(nlohmann_json REQUIRED)

# ceres
find_package(Ceres REQUIRED)

# own lib
add_subdirectory(lib)

# AddressSanitizer

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# ---- Executable targets ----

add_executable(segtree segtree.cpp)
target_link_libraries(segtree libmath libio)

# -----

add_executable(t t.cpp)
target_link_libraries(t libds libbl)
install(TARGETS t)

# -----
add_executable(ceres_test ceres_test.cpp)
target_link_libraries(ceres_test Ceres::ceres)
install(TARGETS t)


# tests
add_subdirectory(test)