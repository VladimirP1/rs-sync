add_library(libvision vision/calibration.cpp vision/tracking.cpp vision/camera_model.cpp)
target_include_directories(libvision PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(libvision PRIVATE ${OpenCV_LIBS})
target_link_libraries(libvision PRIVATE nlohmann_json::nlohmann_json)

# -----

add_library(libmath INTERFACE)
target_include_directories(libmath INTERFACE ${CMAKE_CURRENT_LIST_DIR})

# -----

add_library(libio io/bb_csv.cpp)
target_include_directories(libio PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(libio PUBLIC libmath)
