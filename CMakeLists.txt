cmake_minimum_required(VERSION 3.5)
project(CPPSim)

# CXX Flags
set(CMAKE_CXX_FLAGS "--std=c++11 -Wno-unused-parameter -Wno-unused-local-typedefs")

# Include directories
include_directories(src lib)

# Subdirectories
add_subdirectory(src)

# Executable
add_executable(${PROJECT_NAME}
               src/BaseController.cpp
               src/BaseQuadEstimator.cpp
               src/main.cpp
               src/QuadControl.cpp
               src/QuadEstimatorEKF.cpp
               src/Trajectory.cpp)

target_link_libraries(${PROJECT_NAME}
                      Drawing
                      Math
                      MavlinkNode
                      Simulation
                      Utility
                      glut
                      GLU
                      GL
                      pthread)
