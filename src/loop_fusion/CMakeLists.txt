cmake_minimum_required(VERSION 3.16)
project(loop_fusion)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(ament_cmake REQUIRED)
find_package(std_srvs REQUIRED)
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

add_executable(loop_fusion_node
src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/brief_extractor.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
)
ament_target_dependencies(loop_fusion_node std_srvs vins_common rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs camera_models cv_bridge tf2_ros ament_index_cpp)
target_link_libraries(loop_fusion_node Eigen3::Eigen Ceres::ceres Threads::Threads ${OpenCV_LIBS})

install(TARGETS loop_fusion_node
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(FILES ../../support_files/brief_k10L6.bin ../../support_files/brief_pattern.yml DESTINATION share/${PROJECT_NAME}/support_files)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_brief_extractor
    test/test_brief_extractor.cpp src/brief_extractor.cpp
    src/ThirdParty/DVision/BRIEF.cpp src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp)
  target_include_directories(test_brief_extractor PRIVATE src)
  target_link_libraries(test_brief_extractor ${OpenCV_LIBS} Threads::Threads)
  target_compile_definitions(test_brief_extractor PRIVATE
    BRIEF_TEST_PATTERN="${CMAKE_CURRENT_SOURCE_DIR}/../../support_files/brief_pattern.yml")
endif()
ament_package()
