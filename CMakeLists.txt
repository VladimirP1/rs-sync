project(rs-sync)

set(BUILD_SHARED_LIBS YES)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

include(thirdparty/conan.cmake)

set( CONAN_REQUIREMENTS
    opencv/4.5.3@demo/testing
    eigen/3.3.8
    ceres-solver/2.0.0
    nlohmann_json/3.10.5
    catch2/2.13.8
    gmp/6.2.1
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

conan_cmake_configure(REQUIRES ${CONAN_REQUIREMENTS}
                      GENERATORS cmake_find_package_multi
                      OPTIONS opencv:with_ffmpeg=True
                      SETTINGS build_type=Release
                    )

set(CMAKE_CONFIGURATION_TYPES Debug Release)
foreach(TYPE ${CMAKE_CONFIGURATION_TYPES})
    conan_cmake_autodetect(settings BUILD_TYPE ${TYPE})
    conan_cmake_install(PATH_OR_REFERENCE .
                        BUILD missing
                        REMOTE conancenter
                        SETTINGS ${settings})
endforeach()

add_library(rtl INTERFACE)
target_include_directories(rtl INTERFACE thirdparty/rtl)

add_subdirectory(src)