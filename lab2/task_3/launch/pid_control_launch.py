from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='task_3',
            executable='pid_controller',
            name='pid'
        )
    ])