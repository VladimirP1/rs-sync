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
target_include_directories(track_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(track_test ${OpenCV_LIBS})
target_link_libraries(track_test libvision)

# -----

add_executable(track_corr track_corr.cpp)
target_include_directories(track_corr PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(track_corr ${OpenCV_LIBS})
target_link_libraries(track_corr libvision libio)

# -----

add_executable(main main.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main libvision)

# -----

add_executable(t t.cpp)
target_include_directories(t PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(t ${OpenCV_LIBS})
target_link_libraries(t libvision)


# --------------------------------------------------------------------------

# add_executable(cv cv.cpp tracking.cpp)
# target_include_directories(cv PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(cv ${OpenCV_LIBS})

# -----

# add_executable(main main.cpp calibration.cpp)
# target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(main ${OpenCV_LIBS})
# target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
