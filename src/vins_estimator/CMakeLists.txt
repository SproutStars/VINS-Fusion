cmake_minimum_required(VERSION 3.16)
project(vins)
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
find_package(OpenCV 4 REQUIRED)
find_package(vins_common REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(camera_models REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Opt-in host tuning. -march=native changes the floating-point instruction mix
# (notably FMA) and caused deterministic divergence on EuRoC V2_02_medium, so
# portable code generation remains the default. Do not ship native binaries.
option(VINS_ENABLE_NATIVE_OPTIMIZATION
  "Build with -march=native (numeric results may change; not for release)" OFF)
if(VINS_ENABLE_NATIVE_OPTIMIZATION)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" VINS_HAS_MARCH_NATIVE)
  if(VINS_HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  else()
    message(WARNING "VINS_ENABLE_NATIVE_OPTIMIZATION requested but -march=native is unsupported")
  endif()
endif()

set(VINS_AMENT_DEPENDENCIES
  vins_common
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  camera_models
  cv_bridge
  tf2_ros
  ament_index_cpp
)

set(VINS_LINK_LIBRARIES
  Eigen3::Eigen
  Ceres::ceres
  Threads::Threads
  ${OpenCV_LIBS}
)

add_library(vins_lib
src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    src/backend/ekf_state.cpp
    src/backend/schur_update.cpp
)
ament_target_dependencies(vins_lib ${VINS_AMENT_DEPENDENCIES})
target_link_libraries(vins_lib ${VINS_LINK_LIBRARIES})
target_include_directories(vins_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:include/vins>)

add_executable(vins_node
src/rosNodeTest.cpp
)
add_executable(kitti_odom_test
src/KITTIOdomTest.cpp
)
add_executable(kitti_gps_test
src/KITTIGPSTest.cpp
)

foreach(vins_target IN ITEMS vins_node kitti_odom_test kitti_gps_test)
  ament_target_dependencies(${vins_target} ${VINS_AMENT_DEPENDENCIES})
  target_link_libraries(${vins_target} ${VINS_LINK_LIBRARIES} vins_lib)
endforeach()

install(TARGETS vins_lib vins_node kitti_odom_test kitti_gps_test
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_runtime_utils test/test_runtime_utils.cpp)
  target_include_directories(test_runtime_utils PRIVATE src)
  target_link_libraries(test_runtime_utils Threads::Threads)
  ament_add_gtest(test_feature_tracker test/test_feature_tracker.cpp)
  target_link_libraries(test_feature_tracker vins_lib)
  ament_target_dependencies(test_feature_tracker vins_common)
  ament_add_gtest(test_solver_timeout test/test_solver_timeout.cpp)
  target_include_directories(test_solver_timeout PRIVATE src)
  target_link_libraries(test_solver_timeout Ceres::ceres)
  ament_add_gtest(test_estimator_recovery test/test_estimator_recovery.cpp)
  target_link_libraries(test_estimator_recovery vins_lib)
  ament_target_dependencies(test_estimator_recovery vins_common)
  ament_add_gtest(test_gftt_pyramid_perf test/test_gftt_pyramid_perf.cpp)
  target_link_libraries(test_gftt_pyramid_perf vins_lib)
  ament_target_dependencies(test_gftt_pyramid_perf vins_common)
  ament_add_gtest(test_feature_index test/test_feature_index.cpp)
  target_link_libraries(test_feature_index vins_lib)
  ament_target_dependencies(test_feature_index vins_common)
  ament_add_gtest(test_pose_manifold test/test_pose_manifold.cpp)
  target_include_directories(test_pose_manifold PRIVATE src)
  target_link_libraries(test_pose_manifold vins_lib Ceres::ceres Eigen3::Eigen)
  ament_target_dependencies(test_pose_manifold vins_common)
  ament_add_gtest(test_ekf_state test/test_ekf_state.cpp)
  target_include_directories(test_ekf_state PRIVATE src)
  target_link_libraries(test_ekf_state vins_lib Eigen3::Eigen Threads::Threads)
  ament_add_gtest(test_schur_update test/test_schur_update.cpp)
  target_include_directories(test_schur_update PRIVATE src)
  target_link_libraries(test_schur_update vins_lib Eigen3::Eigen Threads::Threads)
endif()
ament_package()
