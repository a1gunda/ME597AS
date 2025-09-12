from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='task_2',
            executable='talker',
            name='publisher'
        ),
        Node(
            package='task_2',
            executable='listener',
            name='subscriber'
        ),
        Node(
            package='task_2',
            executable='service',
            name='service'
        ),
        Node(
            package='task_2',
            executable='client',
            name='client'
        )
    ])