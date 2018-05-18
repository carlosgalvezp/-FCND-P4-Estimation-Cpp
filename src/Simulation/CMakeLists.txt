project(Simulation)

add_library(${PROJECT_NAME}
            BaseDynamics.cpp
            magnetometer.cpp
            opticalflow.cpp
            QuadDynamics.cpp
            rangefinder.cpp
            Simulator.cpp)

target_link_libraries(${PROJECT_NAME} Math)
