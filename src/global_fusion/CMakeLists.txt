cmake_minimum_required(VERSION 3.16)
project(global_fusion)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres 2.2 REQUIRED)
find_package(Threads REQUIRED)
find_package(vins_common REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
add_library(vins_geographic STATIC
  ThirdParty/GeographicLib/src/Geocentric.cpp
  ThirdParty/GeographicLib/src/LocalCartesian.cpp
  ThirdParty/GeographicLib/src/Math.cpp)
target_include_directories(vins_geographic PUBLIC ThirdParty/GeographicLib/include)

add_executable(global_fusion_node
src/globalOptNode.cpp
	src/globalOpt.cpp
)
ament_target_dependencies(global_fusion_node vins_common rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs)
target_link_libraries(global_fusion_node Eigen3::Eigen Ceres::ceres Threads::Threads vins_geographic)

install(TARGETS global_fusion_node
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})
ament_package()
