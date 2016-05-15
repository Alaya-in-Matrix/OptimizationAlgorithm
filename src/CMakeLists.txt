cmake_minimum_required(VERSION 3.2.1)
project(optimizer)
set(CMAKE_BUILD_TYPE debug)

set(EXE solver)
set(SRC main.cpp benchmark.cpp obj.cpp optimizer.cpp)

add_executable(${EXE} ${SRC})
set_property(TARGET ${EXE} ${DE_SHARED} ${DE_STATIC} ${TEST} PROPERTY CXX_STANDARD 11)
