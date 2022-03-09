find_package(Armadillo CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

add_executable(gd simple_gd.cpp quat.cpp signal.cpp cv.cpp ndspline.cpp)
target_link_libraries(gd PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp spline tbb)

add_executable(only_plot only_plot.cpp quat.cpp signal.cpp cv.cpp ndspline.cpp)
target_link_libraries(only_plot PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp spline tbb)
