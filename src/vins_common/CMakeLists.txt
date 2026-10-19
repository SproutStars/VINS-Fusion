cmake_minimum_required(VERSION 3.16)
project(vins_common)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_dependencies(rclcpp OpenCV Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_vins_common_utility test/test_utility.cpp)
  target_include_directories(test_vins_common_utility PRIVATE include)
  target_link_libraries(test_vins_common_utility Eigen3::Eigen)

  ament_add_gtest(test_vins_common_tic_toc test/test_tic_toc.cpp)
  target_include_directories(test_vins_common_tic_toc PRIVATE include)
endif()

ament_package()
