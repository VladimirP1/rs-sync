add_library(support quat.cpp signal.cpp cv.cpp minispline.cpp ndspline.cpp backtrack.cpp)
target_include_directories(support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(support PUBLIC ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp TBB::tbb GSL::gsl GSL::gslcblas)