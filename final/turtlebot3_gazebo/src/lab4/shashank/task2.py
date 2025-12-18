#!/usr/bin/env python3

import rclpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import os
import cv2
import sys
import subprocess
import time
import traceback
import heapq as hq
import math

from rclpy.node import Node
from collections import deque
from matplotlib.colors import ListedColormap, BoundaryNorm
from given_functions import Map, Tree, givenNode, RRTStar, AStar # Assuming these are correctly implemented
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PointStamped, PoseStamped, Twist
from std_msgs.msg import Float32 # This should likely be Twist for velocity commands
from gazebo_msgs.msg import ContactsState

# Import other python packages that you think necessary


class Task2(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self):
        super().__init__('task2_node')

        # Variable initialization
        self.map_array = None
        self.bot_state = 0
        self.bot_position = None
        self.start_time = self.get_clock().now()
        self.bot_heading_ang = None # Initialize heading angle
        self.pose_msg = None # Initialize pose message
        self.crash = False # Initialize crash state
        self.start_time = self.get_clock().now()
        self.goal_pose = None
        self.crisis_ctr = np.inf
        self.speed = 0.0
        self.heading = 0.0

        # Subscribers
        self.pos_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.PoseCB, 10)
        # self.pos_sub = self.create_subscription(Odometry, '/odom', self.PoseCB, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.ScanCB, 10)
        self.collision_sub = self.create_subscription(ContactsState, '/bumper_collisions', self.CrashCB, 10)
        self.goalPose_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)

        # Publisher
        self.goal_pub = self.create_publisher(PointStamped, '/frontier_point', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.path_pub = self.create_publisher(Path, '/movement_path', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) # Corrected type

        # get the map
        self.map = Map(map_name='map')
        self.map_dat = self.map.info
        if self.map_array == None:
            if self.map_dat.width > 0 and self.map_dat.height > 0 and len(self.map.data) > 0:
                self.map_array = np.array(self.map.data, dtype=np.int16)
                self.map_array = np.reshape(self.map_array, (self.map_dat.height, self.map_dat.width))
                self.sminf_map_array = inflate_obstacles(self.map_array, 4)
                self.map_array = inflate_obstacles(self.map_array, 8)
                self.og_map_array = self.map_array
                self.get_logger().info('Map processed and inflated', once=True) 
            else:
                self.get_logger().warn('Received empty or invalid map data')
        
        
    def PoseCB(self, msg):
        self.pose_msg = msg.pose
        self.bot_position = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y))
        self.bot_position = self.getClosestPoint(current_position_ij=world2map(self.map, self.bot_position))

        heading_quat = msg.pose.pose.orientation
        x, y, self.bot_heading_ang = quaternion_to_heading_vector(heading_quat)
        self.bot_heading_vec = (x,y)
    
    def ScanCB(self, msg: LaserScan):
        if (self.map is None or self.bot_position is None):
            return
        # 1) compute angles for each measurement
        angles = np.arange(msg.angle_min,
                           msg.angle_max + msg.angle_increment/2,
                           msg.angle_increment)
        ranges = np.array(msg.ranges)

        length = len(ranges)
        mid_index = length // 2
        back_lr = ranges[mid_index-2:mid_index+2]
        self.laserscan_x_p = back_lr.min()

        # filter out invalid ranges
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        angles = angles[valid]
        og_range = ranges
        ranges = ranges[valid]

        front_right = ranges[0:2]
        front_left = ranges[-2:]
        self.laserscan_x = min(front_left.min(), front_right.min())

        # in‐robot‐frame points
        xs_r = ranges * np.cos(angles)
        ys_r = ranges * np.sin(angles)        

        smallest_r = np.inf
        for i, r in enumerate(ranges):
            if r < smallest_r:
                smallest_r = r
                smallest_ang = angles[i]
                smallest_i = i

        try:
            self.away_from_wall = tuple(np.array([[np.cos(smallest_ang), -np.sin(smallest_ang)],
                                      [np.sin(smallest_ang), np.cos(smallest_ang)]]) @ np.array(self.bot_heading_vec))
        except:
            self.away_from_wall = self.bot_heading_vec
        
        bin_position = tuple(np.array(self.bot_position) + (0.25 + self.laserscan_x) * np.array(self.bot_heading_vec))
        nin_ij = world2map(self.map, bin_position)
        self.get_logger().info(f"GRADE THIS LAST!! IS NOT FULLY WORKING!!")

        # try:
        #     self.get_logger().info(f"The map of the binpos is {self.sminf_map_array[nin_ij[1],nin_ij[0]]}")
        # except:
        #     self.get_logger().info(f"Map shape is {np.shape(self.sminf_map_array)} , {nin_ij}")

        bin_position = self.getClosestPoint(world2map(self.map, bin_position))
        binflation = 8

        detection_range = 0.6
        if (any(og_range[i] < detection_range for i in range(15)) or any(og_range[i] < detection_range for i in range(-15, -1))) and any(og_range[i] < detection_range for i in range(mid_index-15, mid_index+15)):
            self.get_logger().info("BOXED IN")
            self.move_ttbot(speed=0.0, heading=0.25)
            time.sleep(2)
            return

        elif (any(og_range[i] <  detection_range for i in range(15)) or any(og_range[i] < detection_range for i in range(-15, -1))):
            self.get_logger().info("CRASH  Immenent")
            if self.laserscan_x_p < 0.9:
                self.move_ttbot(speed=0.0, heading=-0.25)
            else:
                self.move_ttbot(speed=-0.25, heading=0.005)
            time.sleep(0.7)
            self.StopNow()
            time.sleep(0.2)
            return
        
        elif any(og_range[i] < detection_range for i in range(mid_index-15, mid_index+15)):
            self.get_logger().info("CRASH IMMINENT -Back")
            if self.laserscan_x < 0.9:
                self.move_ttbot(speed=0.0, heading=-0.25)
            else:
                self.move_ttbot(speed=0.25, heading=0.005)
            time.sleep(0.7)
            self.StopNow()
            time.sleep(0.2)
            return

        if np.abs(self.heading) < 0.1:
            look_ang = 6
        else:
            look_ang = 10
        
        # self.get_logger().info(f'Laserscan x = {self.laserscan_x}, any {any(og_range[i] < 1.2 for i in range(look_ang)) or any(og_range[i] < 1.2 for i in range(-look_ang, -1))}, speed{self.speed}, {self.speed>0}')
        if (any(og_range[i] < 1.2 for i in range(look_ang)) or any(og_range[i] < 1.2 for i in range(-look_ang, -1))) and (self.speed > -0.1):
            self.get_logger().info("CRASH  -- Making bin")
            if self.laserscan_x_p < 0.7:
                self.move_ttbot(speed=0.0, heading=-0.25)
            else:
                self.move_ttbot(speed=-0.25, heading=0.005)
            time.sleep(0.4)
            self.StopNow()
            time.sleep(0.2)
            if self.crisis_ctr < 3:
                self.crisis_ctr += 1
                return 
            try:
                if self.sminf_map_array[nin_ij[1],nin_ij[0]] > 10:
                    return
            except:
                return
            
            self.bot_state = -1
            self.crisis_ctr = 0
            x1, y1 = world2map(self.map, bin_position)
            
            self.map_array[y1-binflation:y1+binflation, x1-binflation:x1+binflation] = 101
            self.sminf_map_array[y1-binflation:y1+binflation, x1-binflation:x1+binflation] = 101
            self.get_logger().info(f"{(y1-binflation,y1+binflation, x1-binflation,x1+binflation)}")
            self.bot_state = -1
            self.DispMap(map_array=self.map_array)
        

        # 3) world -> map indices, mark occupied
        # for x, y in zip(xs_w, ys_w):
        #     ij = world2map(self.map, (x, y))
        #     if ij is not None:
        #         i, j = ij
        #         # ensure index in bounds
        #         # if 0 <= i < self.map_dat.height and 0 <= j < self.map_dat.width:
        #         #     self.map_array[i, j] = 101
        #         for dx in range(-3, 4):  # -3 to 3
        #             for dy in range(-3, 4):
        #                 if abs(dx) + abs(dy) <= 5:  # optional: restrict to within a certain radius (Manhattan)
        #                     ni, nj = i + dx, j + dy
        #                     if 0 <= ni < self.map_dat.height and 0 <= nj < self.map_dat.width:
        #                         self.map_array[ni, nj] = 101

    def CrashCB(self,msg=ContactsState()):
        if len(msg.states) == 0:
            self.crash = False
        else:
            self.crash = True
            self.crisis_ctr = 1e6

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose_data = data
        self.goal_pose = (self.goal_pose_data.pose.position.x, self.goal_pose_data.pose.position.y)
        self.get_logger().info(f'Goal_pose recieved: {self.goal_pose}')
        self.bot_state = 2

        map_pose = world2map(self.map, self.goal_pose)
        if self.map_array[map_pose[1], map_pose[0]] != 0:
            self.goal_pose = self.getClosestPoint(map_pose)

        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "map"  # Use "map" frame for map coordinates
        point_msg.point.x = self.goal_pose[0]
        point_msg.point.y = self.goal_pose[1]
        point_msg.point.z = 0.0 # Assuming 2D

        self.goal_pub.publish(point_msg)
        self.move_ttbot(0.0,0.0)
        

    def getClosestPoint(self, current_position_ij):
        """
        Given the map array, and the current position, this function finds the closest unexplored territory
        i,j has been incorrectly swapped in this code but I swapped it in return so it all works out
        """
        curr_i = current_position_ij[0]
        curr_j = current_position_ij[1]
        max_i = self.map_dat.width
        max_j = self.map_dat.height

        min_dist_sq = np.inf
        min_threshold = 0
        front_i = curr_i
        front_j = curr_j

        for i in range(1, max_i):
            for j in range(1, max_j):
                if self.map_array[j, i] < 0:
                    continue
                elif self.map_array[j, i] > 10:
                    continue

                try:
                    dist_sq = ((i - curr_i)**2 + (j - curr_j)**2)
                    if dist_sq < min_dist_sq and dist_sq >= min_threshold:
                        front_i = i
                        front_j = j
                        min_dist_sq = dist_sq
                except:
                    self.get_logger().info(f'Shape of map array is {np.shape(self.map_array)} (j, i) is {(j,i)}')

        closest_point = map2world(self.map, (front_i, front_j))

        return closest_point
    
    ##################################
    # Path following
    #################################

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired linear speed.
        @param  heading   Desired angular velocity (yaw rate).
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = float(speed) # Ensure float type
        cmd_vel.angular.z = float(heading) # Ensure float type

        self.speed = float(speed)
        self.heading = float(heading)

        if self.crash:
            if self.laserscan_x > self.laserscan_x_p:
                cmd_vel.linear.x = 0.2
            elif self.laserscan_x < self.laserscan_x_p:
                cmd_vel.linear.x = -0.2
            else:
                cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.1
            self.get_logger().info("CRASHED")

        try:
            self.cmd_vel_pub.publish(cmd_vel)
            if self.crash:
                time.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f"Failed to publish command velocity: {e}")

    def StopNow(self):       
        self.currently_driving = 0
        self.move_ttbot(0.0,0.0)
        self.get_logger().info('Stopping!!')

    def path_follower(self, path, current_goal_pose=None):
        """! Path follower.
        @return speed, heading         Control commands for speed and heading.
        """
        if current_goal_pose is None:
            current_goal_pose = path[self.path_idx]
            skip_indexing = False
        else:
            skip_indexing = True

        # PID gains for velocity (distance control)
        kp_vel = 1.1   # proportional gain
        ki_vel = 0.5  # integral gain
        kd_vel = -0.25  # derivative gain

        # PID gains for heading (angle control)
        kp_head = 0.4 # proportional gain
        ki_head = 0.0 # integral gain
        kd_head = 0.0 # derivative gain

        min_speed = 0.01
        max_speed = 0.35 ### IMPORTANT CHECK HERE

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
        heading_threshold_rad = np.deg2rad(20)
        weight = np.clip(1.0 - (abs(heading_error) / heading_threshold_rad), 0.0, 1.0)
        speed = np.clip(speed, min_speed, max_speed) * weight
        heading = heading * (1.0 - weight)

        # Increment the goal index if very close to the current goal
        
        if distance_to_goal < 0.25:
            speed = 0.0
            heading = 0.0
            if not skip_indexing:
                self.path_idx += 1
                self.get_logger().info("increment")
                if self.path_idx > len(path):
                    self.get_logger().info("End of path")
                    self.StopNow()

        return speed, heading

    #########################
    # Auxilary functions
    #########################

    def DispMap(self, map_array, x1=None, y1=None, x2=None, y2=None, stopTime=3):
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

    ###############################
    # Navigation
    ###############################
    def getPath(self, use_astar=1):
        start_map_pos = world2map(self.map, self.bot_position)
        goal_map_pos = world2map(self.map, self.goal_pose)
        i,j = goal_map_pos

        if start_map_pos is None or start_map_pos[0] < 0 or start_map_pos[0] >= self.map_dat.width or \
                start_map_pos[1] < 0 or start_map_pos[1] >= self.map_dat.height:
                 self.get_logger().error(f"Invalid start position in map coordinates: {start_map_pos}")
                 return None

        if i < 0 or i >= self.map_dat.width or j < 0 or j >= self.map_dat.height:
            self.get_logger().error(f"Invalid frontier goal position in map coordinates: ({i},{j})")
            return None
        
        if use_astar:
            a_star = AStar(start_pos_vec=start_map_pos,
                                end_pos_vec=(i,j),
                                occupancy_map=self.map_array,
                                visualize=False)
            
            path = a_star.find_path_smooth() # This method should return a list of map coordinates (i, j)
            if path == None:
                self.get_logger().info(f'A* Path failed from {start_map_pos} to {(i,j)}')
                self.bot_state=0
                self.DispMap(map_array=self.map_array, x1=start_map_pos[0], y1 = start_map_pos[1], x2=i, y2 = j, stopTime=1.5)
                return None
            
            self.get_logger().info(f'A* generated a path with {len(path)} points.')
            self.path_idx = 0 # Reset path index for the new path
        
        else:
            self.DispMap(map_array=self.map_array, x1 = i, y1 = j, x2 = start_map_pos[0], y2 = start_map_pos[1])
            rrt_star = RRTStar(start_pos_vec=start_map_pos,
                                end_pos_vec=(i,j),
                                occupancy_map=self.map_array, # Pass the numpy array
                                step_size=5.0,
                                search_radius=10.0,
                                goal_bias=0.25,
                                max_iterations=1000,
                                visualize=True)
            
            path = rrt_star.find_path_smooth()
            if path == None:
                self.get_logger().info(f'RRT* Path failed in 2000 iterations from {start_map_pos} to {(i,j)}')
                self.DispMap(temp_map_array=self.map_array, x1=start_map_pos[0], y1 = start_map_pos[1], x2=i, y2 = j, stopTime=15)
                return None
            
            self.get_logger().info(f'RRT* generated a path with {len(path)} points.')
            self.path_idx = 0

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map" # Path points are in map frame, convert to world for Path message
        self.get_logger().info(f'Path:{path}')
        path_world = []
        for ij in path:
            xy_point = map2world(self.map, ij)
            path_world.append(xy_point)
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "map" # Frame ID for PoseStamped should also be "map" if the path points are in map coordinates
            pose_stamped.pose.position.x = xy_point[0]
            pose_stamped.pose.position.y = xy_point[1]
            pose_stamped.pose.position.z = 0.0 
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)
        return path_world        
        

    ##############################3
    # Main Run Loop
    ###############################

    def run(self):
        self.get_logger().info('Starting Run loop')

        # Give some time for ROS2 to establish connections and receive initial messages
        self.get_logger().info('Waiting for amcl data...')
        while (rclpy.ok() and (self.bot_position is None)):
             rclpy.spin_once(self, timeout_sec=0.5)
             if self.bot_position is None:
                  self.get_logger().info('Waiting for bot position...')
             time.sleep(0.1)


        self.get_logger().info('Received odometry data. Starting navigation.')
        map_array_rest_ctr = 100

        while True:
            rclpy.spin_once(self, timeout_sec=0.2)
            if map_array_rest_ctr < 50:
                map_array_rest_ctr += 1
            else:
                self.map_array = self.og_map_array
            match self.bot_state:
                case 0:
                    self.get_logger().info("Idling waiting for goal pose")
                case -1:
                    self.get_logger().info("CRISIS!! Replanning path")
                    path = self.getPath(use_astar=0)
                    if path is not None:
                        self.bot_state = 2

                case 1:
                    speed, heading = self.path_follower(path=path)
                    self.move_ttbot(speed=speed, heading=heading)

                    if self.path_idx >= len(path):
                        self.bot_state = 0
                        continue

                    current_goal = path[self.path_idx]
                    waypoint = PointStamped()
                    waypoint.header.frame_id = "map"  
                    waypoint.point.z = 0.0 
                    waypoint.point.x = current_goal[0]
                    waypoint.point.y = current_goal[1]
                    self.waypoint_pub.publish(waypoint)
                    # self.get_logger().info(f"Driving to {current_goal}")
                case 2:
                    self.get_logger().info(f"Navigating")
                    path = self.getPath(use_astar=1)
                    if path is not None:
                        self.bot_state = 1


            time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)

    task2 = Task2()

    try:
        task2.run()
    except KeyboardInterrupt:
        task2.get_logger().error(f'An error occurred: {e}')
        task2.get_logger().error(traceback.format_exc())
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()

## Supplementary functions

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
        row_end = min(rows, obs_row + offset + 1) # +1 because slicing is exclusive
        col_start = max(0, obs_col - offset)
        col_end = min(cols, obs_col + offset + 1) # +1 because slicing is exclusive



        # Get the slice of the map to inflate
        map_slice = inflated_map[row_start:row_end, col_start:col_end]


        # Apply the inflation: set values < 100 in the kernel area to 100
        inflated_map[row_start:row_end, col_start:col_end][inflated_map[row_start:row_end, col_start:col_end] < 100] = 100



    return inflated_map

def map2world(given_map, ij):
        """Converts node (map) coordinates (i, j) to world coordinates (x, y)
        using effective parameters from manual mapping."""
        i = ij[0]
        j = ij[1]
        # Effective resolution deduced from manual mapping:
        res = 0.075
        # Effective x-origin so that (0,0) yields x ~ –5.39:
        x_origin_eff = -5.39 - 0.5 * res   # ≈ -5.4275
        # Let the top of the map (node row 0) correspond to y_top = 4.24.
        # Then with image height H, the effective y-origin is:
        H = given_map.map_im.size[1]
        y_origin_eff = 4.24 - (H - 0.5) * res
        x_world = x_origin_eff + (i + 0.5) * res
        y_world = y_origin_eff + (H - j - 0.5) * res
        return (x_world, y_world)

def world2map(given_map, xy):
        x_world = xy[0]
        y_world = xy[1]
        res = 0.075
        x_origin_eff = -5.39 - 0.5 * res  # ≈ -5.4275
        H = given_map.map_im.size[1]
        y_origin_eff = 4.24 - (H - 0.5) * res
        i = int(round((x_world - x_origin_eff) / res - 0.5))
        j = int(round(H - (y_world - y_origin_eff) / res - 0.5))
        return (i,j)


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


if __name__ == '__main__':
    main()
