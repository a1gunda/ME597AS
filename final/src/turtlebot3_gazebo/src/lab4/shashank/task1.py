#!/usr/bin/env python3

import rclpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import os
import sys
import cv2
import subprocess
import time
import traceback
import heapq as hq
import math

from rclpy.node import Node
from collections import deque
from matplotlib.colors import ListedColormap, BoundaryNorm
from given_functions import Map, Tree, givenNode, RRTStar, AStar # Assuming these are correctly implemented
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PointStamped, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32 # This should likely be Twist for velocity commands

# Import other python packages that you think necessary


class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        # Fill in the initialization member variables that you need
        self.currently_driving = 0 # Consider using a boolean
        self.get_logger().info('Initialized node')

        self.pos_sub = self.create_subscription(Odometry, '/odom', self.PoseCB, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.OccupancyCB, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.ScanCB, 10)
        # self.timer = self.create_timer(0.1, self.timer_cb) # Timer is commented out, but timer_cb is also commented out.

        self.frontier_pub = self.create_publisher(PointStamped, '/frontier_point', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.path_pub = self.create_publisher(Path, '/movement_path', 10)
        # self.cmd_vel_pub = self.create_publisher(Float32, '/cmd_vel', 10) # This is likely a bug, cmd_vel should be Twist
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) # Corrected type

        # Initialize map and position variables, as callbacks might not be called immediately
        self.current_map = None
        self.map_array = None
        self.map_dat = None
        self.bot_position = None
        self.bot_heading_vec = None
        self.bot_heading_ang = None
        self.path_idx = 0 # Initialize path index

        # Initialize PID variables
        self.prev_distance_error = 0.0
        self.integral_distance_error = 0.0
        self.prev_heading_error = 0.0
        self.integral_heading_error = 0.0
        self.last_time = self.get_clock().now() # For calculating dt in PID

        #
        self.path_failed_couter = 0


    def OccupancyCB(self, msg):
        self.get_logger().info('Map received', once=True) # Added for debugging
        self.current_map = msg
        self.map_dat = self.current_map.info
        # Check if map data is valid before processing
        if self.map_dat.width > 0 and self.map_dat.height > 0 and len(self.current_map.data) > 0:
            self.map_array = np.array(self.current_map.data, dtype=np.int8) # Specify dtype
            # Reshape correctly based on height and width
            self.map_array = np.reshape(self.map_array, (self.map_dat.height, self.map_dat.width))
            # Ensure inflation happens only after the map is valid
            self.map_array = inflate_obstacles(self.map_array, 14)
            self.get_logger().info('Map processed and inflated', once=True) # Added for debugging
        else:
            self.get_logger().warn('Received empty or invalid map data')
    
    def ScanCB(self, msg):
        if (self.current_map is None or self.bot_position is None):
            return
        # 1) compute angles for each measurement
        angles = np.arange(msg.angle_min,
                           msg.angle_max + msg.angle_increment/2,
                           msg.angle_increment)
        ranges = np.array(msg.ranges)

        length = len(ranges)
        mid_index = length // 2
        self.laserscan_x_p = ranges[mid_index]

        # filter out invalid ranges
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        angles = angles[valid]
        ranges = ranges[valid]

        # in‐robot‐frame points
        xs_r = ranges * np.cos(angles)
        ys_r = ranges * np.sin(angles)
        self.laserscan_x = ranges[0]

        smallest_r = np.inf
        for i, r in enumerate(ranges):
            if r < smallest_r:
                smallest_r = r
                smallest_ang = angles[i]

        try:
            self.away_from_wall = tuple(np.array([[np.cos(smallest_ang), -np.sin(smallest_ang)],
                                      [np.sin(smallest_ang), np.cos(smallest_ang)]]) @ np.array(self.bot_heading_vec))
        except:
            self.away_from_wall = self.bot_heading_vec
            


        if any(ranges[i] < 0.42 for i in range(16)) or any(ranges[i] < 0.42 for i in range(-15, -1)):
            self.get_logger().info("CRASH IMMINENT")
            self.move_ttbot(speed=-0.25, heading=0.055)
            time.sleep(0.1)
            # self.bot_state = -1
            # self.crisis_ctr = 0
        
        if any(ranges[i] < 0.35 for i in range(170, 190)):
            self.get_logger().info("CRASH IMMINENT")
            self.move_ttbot(speed=0.25, heading=0.055)
            time.sleep(0.1)
            # self.bot_state = -1
            # self.crisis_ctr = 0

    def DispMap(self, map_array=None, x1=None, y1=None, x2=None, y2=None, stopTime=3):
            """
            Displays the occupancy grid map using OpenCV.

            Args:
                map_array (numpy.ndarray): The occupancy grid map as a 2D NumPy array.
                x1 (int, optional): X-coordinate to highlight (start). Defaults to None.
                y1 (int, optional): Y-coordinate to highlight (start). Defaults to None.
                x2 (int, optional): X-coordinate to highlight (end). Defaults to None.
                y2 (int, optional): Y-coordinate to highlight (end). Defaults to None.
                stopTime (float, optional): Time to display the map (in seconds). Defaults to 3.
            """
            if map_array is None:
                map_array = self.map_array
                if map_array is None:
                    return

            # Define the colormap as a list of BGR colors (OpenCV uses BGR)
            cmap_bgr = [
                [0, 0, 0],       # Black
                [255, 255, 255], # White
                [0, 0, 255],     # Red
                [255, 0, 0],     # Blue  (OpenCV is BGR)
                [0, 255, 255],     # Yellow
            ]

            # Normalize the map_array to 0-255 for display with the colormap
            # We'll shift the values so they start from 0
            display_map = map_array.copy()
            display_map[display_map == -1] = 0  # Black
            display_map[display_map == 0] = 1    # White
            display_map[display_map == 100] = 2  # Red
            display_map[display_map == 101] = 3  
            display_map[display_map == 102] = 4  
            if x1 is not None and y1 is not None:
                display_map[y1-3:y1+4, x1-3:x1+4] = 3 # Blue
            if x2 is not None and y2 is not None:
                display_map[y2-3:y2+4, x2-3:x2+4] = 4 # Yellow

            display_map = display_map.astype(np.uint8)  # Ensure data type is uint8

            # Apply the colormap
            img_colored = np.zeros((display_map.shape[0], display_map.shape[1], 3), dtype=np.uint8)
            for i in range(display_map.shape[0]):
                for j in range(display_map.shape[1]):
                    img_colored[i, j] = cmap_bgr[display_map[i, j]]

            # Resize for better viewing (optional)
            scale_factor = 1  # Adjust as needed
            img_resized = cv2.resize(img_colored, (display_map.shape[1] * scale_factor, display_map.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

            # Display the image using OpenCV
            cv2.imshow('Occupancy Grid Map', img_resized)
            cv2.waitKey(int(stopTime * 1000))  # Convert stopTime to milliseconds
            cv2.destroyAllWindows()

    def FrontierExploration(self):
        """
        This function takes current postion of the robot and finds the closest unexplored frontier, publishes it
        and and returns the path
        Ouputs:
            complete: whether the exploration is complete (boolean)
            path: path to follow to explore (list of tuples or None)
        """
        # Ensure map and position data are available
        if self.map_array is None or self.bot_position is None or self.current_map is None:
            self.get_logger().warn('Map or bot position not available for frontier exploration')
            return False, None

        i,j,complete = self.getFrontierPoint()

        if complete:
            self.get_logger().info(f'Map observation complete')
            return complete, None

        # Use the correct map data for map2world conversion
        ij = (i,j)
        xy = map2world(self.current_map, ij)
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "map"  # Use "map" frame for map coordinates
        point_msg.point.x = xy[0]
        point_msg.point.y = xy[1]
        point_msg.point.z = 0.0 # Assuming 2D

        self.frontier_pub.publish(point_msg)

        path = None # Initialize path to None
        # Check if a new path needs to be planned
        if self.currently_driving == 0: # Assuming 0 means not driving
             # Ensure all required parameters for RRTStar are correct
             # start_pos_vec should be in map coordinates
             start_map_pos = world2map(self.current_map, self.bot_position)
             self.get_logger().info(f'Planning path to frontier: ({start_map_pos[0]}, {start_map_pos[1]}) --> ({i},{j})')
             # Check if start_map_pos is valid and within map bounds
             if start_map_pos is None or start_map_pos[0] < 0 or start_map_pos[0] >= self.map_dat.width or \
                start_map_pos[1] < 0 or start_map_pos[1] >= self.map_dat.height:
                 self.get_logger().error(f"Invalid start position in map coordinates: {start_map_pos}")
                 return False, None

             # Check if the goal (frontier point) is valid and within map bounds
             if i < 0 or i >= self.map_dat.width or j < 0 or j >= self.map_dat.height:
                 self.get_logger().error(f"Invalid frontier goal position in map coordinates: ({i},{j})")
                 return False, None

             # Instantiate RRTStar with correct parameters
             try:
                rrt_star = RRTStar(start_pos_vec=start_map_pos,
                                end_pos_vec=(i,j),
                                occupancy_map=self.map_array, # Pass the numpy array
                                step_size=5.0,
                                search_radius=10.0,
                                goal_bias=0.25,
                                max_iterations=2000)
                
                a_star = AStar(start_pos_vec=start_map_pos,
                                end_pos_vec=(i,j),
                                occupancy_map=self.map_array)

                path = a_star.find_path() # This method should return a list of map coordinates (i, j)
                self.path_idx = 0 # Reset path index for the new path

                if self.path_failed_couter >= 3:
                    if self.laserscan_x > 1.35:
                        self.move_ttbot(0.3, 0.0)
                    elif self.laserscan_x_p > 1.35:
                        self.move_ttbot(-0.3, 0.0)
                    else:
                        self.move_ttbot(-0.08, -0.3)
                    time.sleep(1)
                    self.path_failed_couter = 0
                    self.StopNow()
                    # self.DispMap()

                if path is None:
                    if self.path_failed_couter < 3:
                        self.get_logger().warn('A* failed to generate a path.')
                        self.path_failed_couter += 1

                    # Consider what to do if no path is found. Maybe mark this frontier as unreachable?
                else:
                    self.get_logger().info(f'A* generated a path with {len(path)} points.')
                    # Publish the generated path for visualization
                    path_msg = Path()
                    path_msg.header.stamp = self.get_clock().now().to_msg()
                    path_msg.header.frame_id = "map" # Path points are in map frame, convert to world for Path message
                    self.get_logger().info(f'Path:{path}')
                    for ij in path:
                        xy_point = map2world(self.current_map, ij)
                        pose_stamped = PoseStamped()
                        pose_stamped.header.stamp = self.get_clock().now().to_msg()
                        pose_stamped.header.frame_id = "map" # Frame ID for PoseStamped should also be "map" if the path points are in map coordinates
                        pose_stamped.pose.position.x = xy_point[0]
                        pose_stamped.pose.position.y = xy_point[1]
                        pose_stamped.pose.position.z = 0.0 # Assuming 2D
                        path_msg.poses.append(pose_stamped)

                    self.path_pub.publish(path_msg)
                    self.get_logger().info(f'Starting driving')
                    self.currently_driving = 1 # Set driving flag if a path is found

             except Exception as e:
                 # Handle the error, maybe return complete=False and path=Noneexcept Exception as e:
                 tb = traceback.format_exc()
                 self.get_logger().error(f"Error during RRT* planning in file: {__file__} at line: {traceback.extract_tb(e.__traceback__)[-1][1]}: {e}\nTraceback:\n{tb}")

        return complete, path

    
    def path_follower(self, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      Pose object containing the current target from the created path. This is different from the global target.
        @return speed, heading         Control commands for speed and heading.
        """
        self.get_logger().info(f'Driving to {current_goal_pose}')
        # PID gains for velocity (distance control)
        kp_vel = 1.5   # proportional gain
        ki_vel = 0.5  # integral gain
        kd_vel = 0  # derivative gain

        # PID gains for heading (angle control)
        kp_head = 1.3  # proportional gain
        ki_head = 0.1 # integral gain
        kd_head = 0.5 # derivative gain

        min_speed = 0.01
        max_speed = 0.5 ### IMPORTANT CHECK HERE

        # Extract positions
        vx = self.bot_position[0]
        vy = self.bot_position[1]
        gx = current_goal_pose[0]
        gy = current_goal_pose[1]

        # Calculate distance to goal
        delx = gx - vx
        dely = gy - vy
        distance_to_goal = np.sqrt(delx**2 + dely**2)

        # Calculate goal heading (angle to the goal)
        heading_goal = np.arctan2(dely, delx)

        # Compute heading error and normalize between -pi and pi
        heading_error = heading_goal - self.bot_heading_ang
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # ----------------- PID control for both speed and heading ----------------- #
        # Initialize persistent variables if they don't exist
        if not hasattr(self, 'prev_distance_error'):
            self.prev_distance_error = distance_to_goal
        if not hasattr(self, 'integral_distance_error'):
            self.integral_distance_error = 0.0
        if not hasattr(self, 'prev_heading_error'):
            self.prev_heading_error = heading_error
        if not hasattr(self, 'integral_heading_error'):
            self.integral_heading_error = 0.0

        dt = 0.1

        # Update integral errors
        self.integral_distance_error += distance_to_goal * dt
        self.integral_heading_error += heading_error * dt

        # Calculate derivative errors
        distance_error_derivative = (distance_to_goal - self.prev_distance_error) / dt
        heading_error_derivative = (heading_error - self.prev_heading_error) / dt

        # Store current errors for next iteration
        self.prev_distance_error = distance_to_goal
        self.prev_heading_error = heading_error

        # PID output for speed (based on distance error)
        speed = (kp_vel * distance_to_goal +
                ki_vel * self.integral_distance_error +
                kd_vel * distance_error_derivative)

        # PID output for heading
        heading = (kp_head * heading_error +
                ki_head * self.integral_heading_error +
                kd_head * heading_error_derivative)

        # ----------------- Optional smoothing based on heading error ----------------- #
        heading_threshold_rad = np.deg2rad(30)
        weight = np.clip(1.0 - (abs(heading_error) / heading_threshold_rad), 0.0, 1.0)
        speed = np.clip(speed, min_speed, max_speed) * weight
        heading = heading * (1.0 - weight)
        self.get_logger().info(f"Heading here is {heading} and error is {heading_error}")

        # Increment the goal index if very close to the current goal
        if distance_to_goal < 0.35:
            speed = 0.0
            heading = 0.0
            self.path_idx += 1
            self.get_logger().info("increment")

        return speed, heading

    def getFrontierPoint(self):
        """
        Given the map array, and the current position, this function finds the closest unexplored territory
        i,j has been incorrectly swapped in this code but I swapped it in return so it all works out
        """
        current_postion = world2map(self.current_map, self.bot_position)
        curr_i = current_postion[0]
        curr_j = current_postion[1]
        max_i = self.map_dat.width
        max_j = self.map_dat.height

        min_dist_sq = np.inf
        # min_threshold = 25
        min_threshold = 8
        front_i = curr_i
        front_j = curr_j

        for i in range(1, max_i):
            for j in range(1, max_j):
                if self.map_array[j, i] < 0:
                    continue
                elif self.map_array[j, i] > 10:
                    continue

                try:
                    if (self.map_array[j-1, i] < 0) or (self.map_array[j, i-1] < 0) or (self.map_array[j+1, i] < 0) or (self.map_array[j, i+1] < 0):
                        dist_sq = np.sqrt((i - curr_i)**2 + (j - curr_j)**2)
                        if dist_sq < min_dist_sq and dist_sq > min_threshold:
                            front_i = i
                            front_j = j
                            min_dist_sq = dist_sq
                except:
                    self.get_logger().info(f'Shape of map array is {np.shape(self.map_array)} (j, i) is {(j,i)}')

        exploration_complete = False
        if front_j == curr_j and front_i == curr_i:
            exploration_complete = True

        return front_i, front_j, exploration_complete
    
    def PoseCB(self, msg):
        self.pose_msg = msg.pose
        self.bot_position = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y))

        heading_quat = msg.pose.pose.orientation
        # Ensure quaternion_to_heading_vector is correctly implemented and returns yaw in radians
        _, _, self.bot_heading_ang = quaternion_to_heading_vector(heading_quat)
        # self.get_logger().info(f'Bot Position: {self.bot_position}, Heading: {np.degrees(self.bot_heading_ang):.2f} deg') # For debugging


    def StopNow(self):
        self.currently_driving = 0
        self.move_ttbot(0.0,0.0)
        self.get_logger().info('Stopping!!')

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired linear speed.
        @param  heading   Desired angular velocity (yaw rate).
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = float(speed) # Ensure float type
        cmd_vel.angular.z = float(heading) # Ensure float type

        try:
            self.cmd_vel_pub.publish(cmd_vel)
        except Exception as e:
            self.get_logger().error(f"Failed to publish command velocity: {e}")


    ##########################3
    # THE RUN
    ##########################
    def run(self):
        """
        Main function of the code
        """
        self.get_logger().info('Starting Run loop')
        complete = False # Initialize complete flag

        # Give some time for ROS2 to establish connections and receive initial messages
        self.get_logger().info('Waiting for map and odometry data...')
        while rclpy.ok() and (self.current_map is None or self.bot_position is None):
             rclpy.spin_once(self, timeout_sec=0.5)
             if self.current_map is None:
                  self.get_logger().info('Waiting for map...')
             if self.bot_position is None:
                  self.get_logger().info('Waiting for bot position...')
             time.sleep(0.1)

        self.get_logger().info('Received initial map and odometry data. Starting exploration.')

        #Conduct a spin so that the map is bigger
        self.move_ttbot(0.0, 1.0)
        time.sleep(5)
        self.StopNow()

        waypoint = PointStamped()
        waypoint.header.frame_id = "map"  # Use "map" frame for map coordinates
        waypoint.point.z = 0.0 # Assuming 2D

        self.get_logger().info(f'Bot position is {self.bot_position[0], self.bot_position[1]} and converted it is {map2world(self.current_map, world2map(self.current_map, self.bot_position))}')

        for i in range(1, 100):
            rclpy.spin_once(self, timeout_sec=0.1)


        # Main exploration loop
        while rclpy.ok() and not complete:
            rclpy.spin_once(self, timeout_sec=0.1) # Process callbacks

            # Only run frontier exploration and planning if not currently driving
            # and map/position data is available
            if self.currently_driving == 0 and self.map_array is not None and self.bot_position is not None:
                 complete, path = self.FrontierExploration()
                 driving_map = self.current_map
                 if complete:
                     self.StopNow()
                     self.get_logger().info("Exploration complete. Exiting run loop.")
                     break # Exit the main loop if exploration is complete


            if self.currently_driving == 1 and path is not None and len(path) > 0:
                #  current_goal, reached_end_of_path = self.get_path_idx(path=path, map=driving_map)
                 if self.path_idx >= len(path):
                    reached_end_of_path = True
                    current_goal = None
                 else:
                    reached_end_of_path = False
                    current_goal = map2world(driving_map, path[self.path_idx])

                 if reached_end_of_path or current_goal is None:
                     self.get_logger().info("Finished following the current path.")
                     self.StopNow() # Stop the bot and reset driving flag
                     path = None # Clear the path so a new one can be planned
                     self.path_idx = 0 # Reset path index
                 else:
                    #  self.driveBot(goal=current_goal)
                    waypoint.point.x = current_goal[0]
                    waypoint.point.y = current_goal[1]
                    self.waypoint_pub.publish(waypoint)

                    rclpy.spin_once(self, timeout_sec=0.1)
                    speed, heading = self.path_follower(current_goal_pose=current_goal)
                    self.move_ttbot(speed=speed, heading=heading)
            elif self.currently_driving == 1 and (path is None or len(path) == 0):
                 # This case should ideally not happen if logic is correct,
                 # but added for robustness.
                 self.get_logger().warn("Driving flag is set but no valid path available.")
                 self.StopNow()

            # This sleep is important to prevent the program from crashing
            time.sleep(0.1)


        self.get_logger().info('Run loop finished.')
        os.system("ffplay -nodisp -autoexit /home/ubuntu/Documents/GitHub/ME597_labs/sim_ws/src/turtlebot3_gazebo/src/lab4/ding-36029.mp3") 
        self.StopNow()
        self.DispMap(stopTime=10)

# Supplementary functions

def map2world(occupancy_grid, ij_tup):
    """
    Converts from map frame (i, j) to world frame (x, y) given the
    map is of type OccupancyGrid.

    Parameters:
        occupancy_grid: OccupancyGrid object
        ij_tup: tuple (i, j) in configuration space (row, column)

    Returns:
        tuple (x, y):
            Coordinates in the world space
    """
    if occupancy_grid is None:
        print("Error in map2world: occupancy_grid is None") # Use print for functions outside class
        return None

    # Map origin in world coordinates (bottom-left corner of the map)
    origin_x = occupancy_grid.info.origin.position.x
    origin_y = occupancy_grid.info.origin.position.y
    resolution = occupancy_grid.info.resolution # meters per cell


    x = origin_x + (ij_tup[0] + 0.5) * resolution # j is column -> x
    y = origin_y + (ij_tup[1] + 0.5) * resolution # i is row -> y



    # If ij_tup is (i, j) where i is row and j is column:
    x_world = occupancy_grid.info.origin.position.x + (ij_tup[0] * resolution) # j is column -> x
    y_world = occupancy_grid.info.origin.position.y + (ij_tup[1] * resolution) # i is row -> y

    # Let's assume ij_tup is (i, j) where i is row (y) and j is column (x)
    x_world_center = occupancy_grid.info.origin.position.x + (ij_tup[0] + 0.5) * resolution
    y_world_center = occupancy_grid.info.origin.position.y + (ij_tup[1] + 0.5) * resolution


    return (x_world_center, y_world_center)


def inflate_obstacles(map_array, k):
    """
    Inflates obstacles in a map array by a k x k square.

    Args:
        map_array (np.ndarray): The map array where -1 is unexplored,
                                 0 is explored and safe, and 100 is explored
                                 and dangerous.
        k (int): The size of the inflation square (must be odd).

    Returns:
        np.ndarray: The map array with inflated obstacles.
    """
    if map_array is None:
        print("Error in inflate_obstacles: map_array is None")
        return None

    rows, cols = map_array.shape
    inflated_map = map_array.copy()

    # Find locations of obstacles (value 100)
    obstacle_locations = np.argwhere(map_array == 100)

    # Calculate the offset for the square kernel
    # For k=5, offset = 2. The square is 2 cells in each direction from the center.
    offset = (k - 1) // 2

    # Iterate through each obstacle location
    for obs_row, obs_col in obstacle_locations:
        # Define the bounds of the inflation square around the obstacle cell
        row_start = max(0, obs_row - offset)
        row_end = min(rows, obs_row + offset + 1) 
        col_start = max(0, obs_col - offset)
        col_end = min(cols, obs_col + offset + 1) 

        # Get the slice of the map to inflate
        map_slice = inflated_map[row_start:row_end, col_start:col_end]

        # Apply the inflation: set values < 100 in the kernel area to 100
        inflated_map[row_start:row_end, col_start:col_end][inflated_map[row_start:row_end, col_start:col_end] < 100] = 100



    return inflated_map

def world2map(occupancy_grid, xy_tup):
    """
    Converts from world frame (x, y) to map frame (i, j) given the
    map is of type OccupancyGrid.

    Parameters:
        occupancy_grid: OccupancyGrid object
            The map representation in the ROS2 framework.
        xy_tup: tuple (x, y)
            Coordinates in the world frame.

    Returns:
        tuple (i, j):
            Grid indices in the map frame (row, column).
            Returns None if the point is outside the map.
    """
    if occupancy_grid is None:
        print("Error in world2map: occupancy_grid is None")
        return None

    xy_tup = np.array(xy_tup)
    origin_x = occupancy_grid.info.origin.position.x
    origin_y = occupancy_grid.info.origin.position.y
    resolution = occupancy_grid.info.resolution  # m/cell

    # Convert world coordinates to grid indices
    # x corresponds to column (j)
    # y corresponds to row (i)

    # Calculate indices relative to the origin
    j_float = (xy_tup[0] - origin_x) / resolution
    i_float = (xy_tup[1] - origin_y) / resolution

    # Round to the nearest integer to get cell indices
    j_int = int(j_float)
    i_int = int(i_float)

    # Get map dimensions
    map_width = occupancy_grid.info.width
    map_height = occupancy_grid.info.height

    # Check if the calculated indices are within map bounds
    if 0 <= i_int < map_height and 0 <= j_int < map_width:
        # Return as (i, j) where i is row and j is column
        return (j_int, i_int)
    else:
        print(f"Warning in world2map: World point {xy_tup} is outside map bounds. Calculated indices: ({i_int}, {j_int}). Map size: ({map_height}, {map_width})")
        return None # Return None if outside map bounds


def quaternion_to_heading_vector(vehicle_ori):
        """
        Converts a quaternion orientation to a heading vector (dx, dy) and yaw angle.

        @param:
            vehicle_ori: Quaternion with orientation data (x, y, z, w).

        @return:
            A tuple (dx, dy, yaw) where (dx, dy) is the heading vector
            and yaw is the heading angle in radians [-pi, pi].
        """
        x = vehicle_ori.x
        y = vehicle_ori.y
        z = vehicle_ori.z
        w = vehicle_ori.w

        # Yaw (z-axis rotation) is the angle needed for 2D navigation
        # Using the formula for yaw from quaternion
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Calculate heading vector (dx, dy) from yaw
        dx = np.cos(yaw)
        dy = np.sin(yaw)

        return dx, dy, yaw


def plot_heatmap(data, title="Heatmap", xlabel="X-axis", ylabel="Y-axis", cmap="viridis", colorbar_label="Value"):
    """
    Plots a heatmap from a NumPy array.

    Parameters:
    - data: 2D NumPy array containing the data to be visualized.
    - title: Title of the heatmap.
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    - cmap: Colormap to use (default is 'viridis').
    - colorbar_label: Label for the colorbar.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar(heatmap, label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        task1.run() # Call your main logic function

    except KeyboardInterrupt:
        task1.get_logger().info('Keyboard interrupt received, shutting down.')
    except Exception as e:
        task1.get_logger().error(f'An error occurred: {e}')
        task1.get_logger().error(traceback.format_exc())
    finally:
        # Ensure the node is destroyed and rclpy is shut down cleanly
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()