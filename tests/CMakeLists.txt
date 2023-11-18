cmake_minimum_required(VERSION 2.8.11)
include(ExternalProject)

set(PROJECTNAME tests)
project(${PROJECTNAME})

find_package(OpenSoT REQUIRED)
find_package(XBotInterface REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp urdf roslib kdl_parser robot_state_publisher)

# compite GTest
find_package(GTest)
if (NOT GTEST_FOUND)
    set(gTestSource "/usr/src/gtest")
    ExternalProject_Add(GTest-ext SOURCE_DIR ${gTestSource}
                                  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external"
                                  INSTALL_COMMAND "")
    set(GTEST_LIB_DIRS "${CMAKE_CURRENT_BINARY_DIR}/external/src/GTest-ext-build/")
    set(GTEST_INCLUDE_DIRS ${gTestSource})
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
endif()

include_directories(${GTEST_INCLUDE_DIRS} ${OpenSoT_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${XBotInterface_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

# add local include
include_directories(${CMAKE_CURRENT_SOURCE_DIR} )
enable_testing()

set(NT_TESTS testNormTask)

SET(TestLibs OpenSoTNormTask ${OpenSoT_LIBRARIES} ${GTEST_BOTH_LIBRARIES} ${catkin_LIBRARIES} ${XBotInterface_LIBRARIES})

ADD_EXECUTABLE(testNormTask tasks/TestNormTask.cpp)
TARGET_LINK_LIBRARIES(testNormTask ${TestLibs})
add_dependencies(testNormTask GTest-ext OpenSoTNormTask)
add_test(NAME opensot_norm_task COMMAND testNormTask)




