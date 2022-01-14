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

add_library(libds ds/sync_context)
target_include_directories(libds PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(libds PUBLIC libmath libvision)
install(TARGETS libds)
