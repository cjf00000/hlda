cmake_minimum_required(VERSION 2.8)
project(hlda)

add_subdirectory(gflags)
set(CMAKE_CXX_FLAGS "-march=native -g -Wall -std=c++11")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

include_directories(hlda/common)
include_directories(hlda/model)
#include_directories(hlda)

add_subdirectory(hlda)
add_subdirectory(hlda_test)
