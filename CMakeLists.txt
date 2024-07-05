cmake_minimum_required(VERSION 3.7.0)

if(WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE INTERNAL "" FORCE)
endif()

project(deconv LANGUAGES C CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C11
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# force MSVC to use UTF8
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

set(SRC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
include_directories(${SRC_INCLUDE_DIR})

add_library(psf ${SRC_INCLUDE_DIR}/psf/psf.cpp ${SRC_INCLUDE_DIR}/psf/psf.h)

find_package(OpenCV REQUIRED)

add_executable(psf_test ${SRC_DIR}/psf_test.cpp)
target_link_libraries(psf_test
    PRIVATE psf
    PRIVATE ${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME} ${SRC_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME}
    PRIVATE ${OpenCV_LIBS}
)
