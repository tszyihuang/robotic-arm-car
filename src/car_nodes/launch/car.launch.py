"""Start all five nodes; executing a mission is a separate plan.py call."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    dry_run = ParameterValue(LaunchConfiguration('dry_run'), value_type=bool)
    params = LaunchConfiguration('params_file')
    config = str(Path(get_package_share_directory('car_nodes')) / 'config/car.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='car'),
        DeclareLaunchArgument('dry_run', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=config),
        *[Node(package='car_nodes', executable=name + '_node', namespace=namespace,
               output='screen', parameters=[params, {'dry_run': dry_run}])
          for name in ('sensor', 'base', 'arm', 'vision', 'plan')],
    ])
