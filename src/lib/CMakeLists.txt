add_library(libvision vision/calibration.cpp vision/tracking.cpp vision/camera_model.cpp vision/utils.cpp)
target_include_directories(libvision PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(libvision PUBLIC opencv::opencv)
target_link_libraries(libvision PRIVATE nlohmann_json::nlohmann_json)
install(TARGETS libvision)

# -----

add_library(libmath INTERFACE)
target_include_directories(libmath INTERFACE ${CMAKE_CURRENT_LIST_DIR})
install(TARGETS libmath)

# -----

add_library(libio io/bb_csv.cpp)
target_include_directories(libio PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(libio PUBLIC libmath)
install(TARGETS libio)

# -----

add_library(libds INTERFACE)
target_include_directories(libds INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(libds INTERFACE libmath libvision)
install(TARGETS libds)

# -----

add_library(libbl bl/async_frame_loader.cpp)
target_include_directories(libbl PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(libbl PUBLIC libds)
install(TARGETS libbl)