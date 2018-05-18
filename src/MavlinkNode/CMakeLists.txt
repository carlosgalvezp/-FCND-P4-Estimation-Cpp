project(MavlinkNode)

add_library(${PROJECT_NAME}
            MavlinkNode.cpp
            MavlinkTranslation.cpp
            PracticalSocket.cpp)
