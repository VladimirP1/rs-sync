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

# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# ---- Executable targets ----

add_executable(segtree segtree.cpp)
target_link_libraries(segtree libmath libio)

# -----

add_executable(track_test track_test.cpp)
target_link_libraries(track_test libvision)
install(TARGETS track_test)

# -----

add_executable(track_corr track_corr.cpp)
target_link_libraries(track_corr libvision libio)
install(TARGETS track_corr)

# -----

add_executable(main main.cpp)
target_link_libraries(main libvision)
install(TARGETS main)

# -----

add_executable(t t.cpp)
target_link_libraries(t libds)
install(TARGETS t)

# -----
add_executable(ceres_test ceres_test.cpp)
target_link_libraries(ceres_test Ceres::ceres)
install(TARGETS t)


# tests
add_subdirectory(test)


# --------------------------------------------------------------------------

# add_executable(cv cv.cpp tracking.cpp)
# target_include_directories(cv PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(cv ${OpenCV_LIBS})

# -----

# add_executable(main main.cpp calibration.cpp)
# target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(main ${OpenCV_LIBS})
# target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
