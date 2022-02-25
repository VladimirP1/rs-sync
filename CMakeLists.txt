set(CMAKE_TOOLCHAIN_FILE ext/vcpkg/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

project(rs-sync)

add_library(telemetry_parser_cpp SHARED IMPORTED GLOBAL)
set_target_properties(
    telemetry_parser_cpp PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/target/release/libtelemetry_parser_cpp.so
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/rust/telemetry-parser-cpp/inc
        IMPORTED_NO_SONAME TRUE
)

add_subdirectory(ext/spline)

add_subdirectory(src)