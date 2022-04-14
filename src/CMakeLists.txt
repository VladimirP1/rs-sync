find_package(Armadillo CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(GSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_subdirectory(support)
add_subdirectory(common)

add_executable(rssync_json rssync_json.cpp)
target_link_libraries(rssync_json PRIVATE support common nlohmann_json::nlohmann_json)

add_executable(gd_gsl gd_and_gsl.cpp)
target_link_libraries(gd_gsl PRIVATE support common)

add_executable(gd0 stupid_gd.cpp)
target_link_libraries(gd0 PRIVATE support common)

add_executable(only_plot only_plot.cpp)
target_link_libraries(only_plot PRIVATE support common)

add_executable(test_cost test/test_cost.cpp)
target_link_libraries(test_cost PRIVATE support common)

add_executable(sync_suggest sync_suggest.cpp)
target_link_libraries(sync_suggest PRIVATE support common)

add_executable(histogram histogram.cpp)
target_link_libraries(histogram PRIVATE  ${OpenCV_LIBRARIES})

add_executable(problem_dump problem_dump.cpp)
target_link_libraries(problem_dump PRIVATE support common)
