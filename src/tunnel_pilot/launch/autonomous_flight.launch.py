import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    ydlidar_pkg_dir = get_package_share_directory('ydlidar_ros2_driver')

    lidar_launch_path = os.path.join(
        ydlidar_pkg_dir, 'launch', 'dual_lidar.launch.py')

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(lidar_launch_path)
    )

    cnn_node = Node(
        package='tunnel_pilot',
        executable='cnn_navigator',
        name='cnn_navigator_node',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        lidar_launch,
        cnn_node
    ])
