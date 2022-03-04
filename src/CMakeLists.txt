find_package(Armadillo CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

add_executable(main main.cpp quat.cpp signal.cpp cv.cpp ndspline.cpp)
target_link_libraries(main PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} telemetry_parser_cpp spline)


add_executable(arma arma.cpp)
target_link_libraries(arma PRIVATE ${ARMADILLO_LIBRARIES})


add_executable(c ceres.cpp quat.cpp signal.cpp cv.cpp ndspline.cpp)
target_link_libraries(c PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBRARIES} Ceres::ceres telemetry_parser_cpp spline)

