find_package(Armadillo CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

add_executable(gd simple_gd.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(gd PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb)

add_executable(gd0 stupid_gd.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(gd0 PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb)

add_executable(only_plot only_plot.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(only_plot PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb)

# add_executable(minispline minispline.cpp)
# target_link_libraries(minispline PRIVATE ${ARMADILLO_LIBRARIES} spline)
