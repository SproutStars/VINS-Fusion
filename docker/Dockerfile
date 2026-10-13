FROM ros:jazzy-ros-base
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-colcon-common-extensions \
    libceres-dev libeigen3-dev libopencv-dev libboost-all-dev \
    ros-jazzy-cv-bridge ros-jazzy-tf2-ros ros-jazzy-rviz2 \
    ros-jazzy-ament-cmake-gtest && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /ws
COPY src /ws/src
COPY support_files/brief_k10L6.bin support_files/brief_pattern.yml /ws/support_files/
RUN source /opt/ros/jazzy/setup.bash && \
    MAKEFLAGS=-j2 colcon build --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    sed -i '/exec "$@"/i source /ws/install/setup.bash' /ros_entrypoint.sh
CMD ["ros2", "launch", "vins", "vins_mono_imu_launch.py"]
