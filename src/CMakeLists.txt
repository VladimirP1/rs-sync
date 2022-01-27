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
target_link_libraries(segtree libmath libio libbl Eigen3::Eigen)

# -----

add_executable(t t.cpp)
target_link_libraries(t libds libbl)
install(TARGETS t)

# -----
add_executable(ceres_test2 ceres_test2.cpp)
target_link_libraries(ceres_test2 Ceres::ceres opencv::opencv)
install(TARGETS ceres_test2)

# -----
add_executable(normal_fit normal_fit.cpp)
target_link_libraries(normal_fit  opencv::opencv)
install(TARGETS normal_fit)

# tests
add_subdirectory(test)