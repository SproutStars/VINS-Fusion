# help for checking input parameters.
ros2 run camera_models Calibrations --help

# example pinhole model.
ros2 run camera_models Calibrations -w 12 -h 8 -s 80 -i calibrationdata --camera-model pinhole

# example mei model.
ros2 run camera_models Calibrations -w 12 -h 8 -s 80 -i calibrationdata --camera-model mei
