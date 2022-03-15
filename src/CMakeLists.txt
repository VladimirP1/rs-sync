find_package(Armadillo CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(GSL REQUIRED)

add_executable(gd_gsl gd_and_gsl.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(gd_gsl PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb GSL::gsl GSL::gslcblas)

add_executable(gd simple_gd.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(gd PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb)

add_executable(gd0 stupid_gd.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(gd0 PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb)

add_executable(only_plot only_plot.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(only_plot PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb)

add_executable(test_cost test/test_cost.cpp quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_link_libraries(test_cost PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb GSL::gsl GSL::gslcblas)

# add_executable(minispline minispline.cpp)
# target_link_libraries(minispline PRIVATE ${ARMADILLO_LIBRARIES} spline)
