cmake_minimum_required(VERSION 3.8)
project(car_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteCommand.action"
  "action/RunTasks.action"
  "msg/Boundary.msg"
  "msg/Objects.msg"
  "msg/Encoders.msg"
  "msg/ImuFeedback.msg"
  DEPENDENCIES std_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
