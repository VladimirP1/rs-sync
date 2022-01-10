add_library(libvision vision/calibration.cpp vision/tracking.cpp)
target_include_directories(libvision PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(libvision PRIVATE ${OpenCV_LIBS})
target_link_libraries(libvision PRIVATE nlohmann_json::nlohmann_json)

# -----

add_library(libmath INTERFACE)
target_include_directories(libmath INTERFACE ${CMAKE_CURRENT_LIST_DIR})