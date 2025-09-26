from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # --- Get package directories ---
    turtlebot4_nav_dir = get_package_share_directory('turtlebot4_navigation')
    turtlebot4_viz_dir = get_package_share_directory('turtlebot4_viz')

    # --- Full paths to the target launch files ---
    slam_launch_file = os.path.join(turtlebot4_nav_dir, 'launch', 'slam.launch.py')
    view_robot_launch_file = os.path.join(turtlebot4_viz_dir, 'launch', 'view_robot.launch.py')

    # --- Include the SLAM launch file ---
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch_file)
    )

    # --- Include the RViz visualization launch file ---
    view_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(view_robot_launch_file)
    )

    # --- Combine into one LaunchDescription ---
    return LaunchDescription([
        slam_launch,
        view_robot_launch
    ])