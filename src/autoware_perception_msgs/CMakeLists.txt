cmake_minimum_required(VERSION 3.0.2)
project(autoware_perception_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
  sensor_msgs
  uuid_msgs
)

add_message_files(
  DIRECTORY msg/traffic_light_recognition
  FILES
    LampState.msg
    TrafficLightRoi.msg
    TrafficLightRoiArray.msg
    TrafficLightState.msg
    TrafficLightStateArray.msg
    TrafficLightStateStamped.msg
)



## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
    uuid_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
    geometry_msgs
    sensor_msgs
    uuid_msgs
)
