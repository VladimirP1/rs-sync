set(BUILD_SHARED_LIBS ON)

include(thirdparty/conan.cmake)


set( CONAN_REQUIREMENTS
    eigen/3.3.8
    ceres-solver/2.0.0
    nlohmann_json/3.10.5
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

conan_cmake_configure(REQUIRES ${CONAN_REQUIREMENTS}
                      GENERATORS cmake_find_package
                    )
conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
                    BUILD missing
                    REMOTE conancenter
                    SETTINGS ${settings})

add_subdirectory(src)