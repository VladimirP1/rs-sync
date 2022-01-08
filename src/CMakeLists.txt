project(rs_sync)

# opencv
find_package(OpenCV REQUIRED)

# nlohmann
include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


# -----

add_executable(track_test track_test.cpp tracking.cpp calibration.cpp)

target_include_directories(track_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(track_test ${OpenCV_LIBS})
target_link_libraries(track_test PRIVATE nlohmann_json::nlohmann_json)


# -----

add_executable(cv cv.cpp tracking.cpp)

target_include_directories(cv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cv ${OpenCV_LIBS})

# -----

add_executable(main main.cpp calibration.cpp)


target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
