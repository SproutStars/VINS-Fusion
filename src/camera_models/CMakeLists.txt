cmake_minimum_required(VERSION 3.16)
project(camera_models)
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
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

add_executable(Calibrations
src/intrinsic_calib.cc
)
target_link_libraries(Calibrations PRIVATE camera_models)

add_library(camera_models
src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/PinholeFullCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc
)
target_include_directories(camera_models PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include> ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_models PUBLIC Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS} yaml-cpp Boost::filesystem Boost::program_options Boost::system)

install(TARGETS camera_models EXPORT camera_modelsTargets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(TARGETS Calibrations RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(camera_modelsTargets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(camera_models)
ament_export_dependencies(Eigen3 Ceres OpenCV yaml-cpp Boost)
ament_package(CONFIG_EXTRAS cmake/camera_models-dependencies.cmake)
