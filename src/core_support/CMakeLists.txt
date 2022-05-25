add_library(rssync_coresupport quat.cpp signal.cpp minispline.cpp ndspline.cpp backtrack.cpp panic.cpp)
target_include_directories(rssync_coresupport INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rssync_coresupport PUBLIC ${ARMADILLO_LIBRARIES} TBB::tbb)