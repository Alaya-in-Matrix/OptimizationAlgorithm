cmake_minimum_required(VERSION 3.2.1)
project(optimizer)
set(CMAKE_BUILD_TYPE debug)

set(EXE solver)
set(SRC main.cpp 
        benchmark.cpp 
        obj.cpp 
        line_search.cpp
        optimizer_1d.cpp 
        linear_algebra.cpp 
        multi_dim_optimizer.cpp)
include_directories(eigen/include/eigen3)
add_executable(${EXE} ${SRC})
add_executable(line_search line_search.cpp 
                           test_line_search.cpp
                           linear_algebra.cpp
                           optimizer_1d.cpp
                           benchmark.cpp
                           obj.cpp)
set_property(TARGET ${EXE} line_search PROPERTY CXX_STANDARD 11)

option(DEBUG_OPTIMIZER "Debug macro for optimizer" OFF)
option(WRITE_LOG "Write log during optimization" ON)
if(DEBUG_OPTIMIZER)
    message(STATUS "Open debug marcro")
    add_definitions(-DDEBUG_OPTIMIZER)
else()
    message(STATUS "Close debug marcro")
    remove_definitions(-DDEBUG_OPTIMIZER)
endif()
if(WRITE_LOG)
    message(STATUS "Write log during optimization")
    add_definitions(-DWRITE_LOG)
else()
    message(STATUS "No log during optimization")
    remove_definitions(-DWRITE_LOG)
endif()
