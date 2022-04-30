find_package(Armadillo CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_subdirectory(core_support)
add_subdirectory(core)

add_executable(core_testcode core_testcode.cpp)
target_link_libraries(core_testcode PRIVATE rssync_core rssync_coresupport telemetry_parser_cpp ${OpenCV_LIBRARIES} nlohmann_json::nlohmann_json)
