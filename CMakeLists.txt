cmake_minimum_required(VERSION 3.0.2)
project(iris_vision)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  actionlib
  cmake_modules
  cv_bridge
  diagnostic_aggregator
  diagnostic_msgs
  dynamic_reconfigure
  geometry_msgs
  image_transport
  rosbridge_server
  roscpp
  roslib
  rosparam
  rospy
  rviz
  sensor_msgs
  std_msgs
  std_srvs
  tf
  topic_tools
  visualization_msgs
  message_generation
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)


catkin_package(
  INCLUDE_DIRS
  LIBRARIES 
  # TODO: go and check what is in the headers
  CATKIN_DEPENDS roscpp
                 roslib
                 std_msgs
                 sensor_msgs
                 actionlib
  DEPENDS Boost

)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(PYTHON_EXECUTABLE /usr/bin/python3.8)

catkin_python_setup()


catkin_install_python(PROGRAMS
  src/ghost_net_inference.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/ghost_net_inference_yolo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/ghost_net_inference_yolov5.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/sensor_nav.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)