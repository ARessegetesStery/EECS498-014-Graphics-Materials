cmake_minimum_required(VERSION 3.10)
project(RayTracing)

set(CMAKE_CXX_STANDARD 17)

add_executable(RayTracing main.cpp Scene.cpp Accel.cpp Math.cpp Ray.cpp)