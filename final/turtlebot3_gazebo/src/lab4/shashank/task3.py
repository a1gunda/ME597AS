#!/usr/bin/env python3

import rclpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import cv2
import os
import time
import traceback
import heapq as hq

from rclpy.node import Node
from given_functions import AStar, Map
from matplotlib.colors import ListedColormap, BoundaryNorm
from cv_bridge import CvBridge

# messages
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped, Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, LaserScan
import rclpy.time


class Task3(Node):
    """
    Environment localization and navigation task.
    You can also inherit from Task 2 node if most of the code is duplicated
    """
    def __init__(self):
        super().__init__('task3_node')
        # Init
        self.bot_position = None
        self.bot_heading_ang = None
        self.bot_state = 0
        self.bridge = CvBridge()
        self.frame = None
        self.astar_fail_ctr = 0
        self.crisis_ctr = 0

        # Subscribers
        self.collision_sub = self.create_subscription(ContactsState, '/bumper_collisions', self.CrashCB, 10)
        self.pos_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.PoseCB, 10)
        self.img_sub = self.create_subscription(Image, '/camera/image_raw', self.ImgCB, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.ScanCB, 10)

        # Publishers
        self.goal_pub = self.create_publisher(PointStamped, '/frontier_point', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.red_pub = self.create_publisher(PointStamped, '/red_pos', 10)
        self.green_pub = self.create_publisher(PointStamped, '/green_pos', 10)
        self.blue_pub = self.create_publisher(PointStamped, '/blue_pos', 10)
        self.path_pub = self.create_publisher(Path, '/movement_path', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) # Corrected type

        # get the map
        self.map = Map(map_name='map')
        self.map_dat = self.map.info
        self.map_array = None
        if self.map_array is None:
            if self.map_dat.width > 0 and self.map_dat.height > 0 and len(self.map.data) > 0:
                self.map_array = np.array(self.map.data, dtype=np.int16)
                self.map_array = np.reshape(self.map_array, (self.map_dat.height, self.map_dat.width))
                self.map_array = inflate_obstacles(self.map_array, 9)
                self.get_logger().info('Map processed and inflated', once=True) 
        else:
            self.get_logger().warn('Received empty or invalid map data')
            return

        self.DispMap(map_array=self.map_array, stopTime=1.5)

        # Waypoint Checklist:
        waypoint_list = [(65,60), (14,30), (12,100), (165,35), (116,10), (180,83)]
        self.waypoint_checklist = dict(zip(waypoint_list, [False] * len(waypoint_list)))

        colors_list = ["red", "green", "blue"]
        self.colors_checklist = dict(zip(colors_list, [False] * len(colors_list)))
        
    ##
    # Subscribers
    ##
    def CrashCB(self,msg=ContactsState()):
        if len(msg.states) == 0:
            self.bot_state = self.bot_state
        else:
            self.bot_state = -1
            self.crisis_ctr = 0

    def PoseCB(self, msg):
        self.pose_msg = msg.pose
        self.bot_position = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y))

        heading_quat = msg.pose.pose.orientation
        x_vec, y_vec, self.bot_heading_ang = quaternion_to_heading_vector(heading_quat)
        self.bot_heading_vec = (x_vec, y_vec)
    
    def ImgCB(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow("Detections", self.frame)
        cv2.waitKey(1)
    
    def ScanCB(self, msg: LaserScan):
        if (self.map is None or self.bot_position is None):
            return
        # 1) compute angles for each measurement
        angles = np.arange(msg.angle_min,
                           msg.angle_max + msg.angle_increment/2,
                           msg.angle_increment)
        ranges = np.array(msg.ranges)

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

        self.away_from_wall = tuple(np.array([[np.cos(smallest_ang), -np.sin(smallest_ang)],
                                      [np.sin(smallest_ang), np.cos(smallest_ang)]]) @ np.array(self.bot_heading_vec))
            


        if any(ranges[i] < 0.35 for i in range(16)) or any(ranges[i] < 0.35 for i in range(-15, -1)):
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

    ##
    # Publishers and path following
    ##
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

    def path_follower(self, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      Pose object containing the current target from the created path. This is different from the global target.
        @return speed, heading         Control commands for speed and heading.
        """
        # PID gains for velocity (distance control)
        kp_vel = 1.5   # proportional gain
        ki_vel = 0.5  # integral gain
        kd_vel = 0  # derivative gain

        # PID gains for heading (angle control)
        kp_head = 0.6  # proportional gain
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
        # self.get_logger().info(f'heading is {np.arctan2(np.sin(heading_error), np.cos(heading_error))}= arctan({np.sin(heading_error)} / {np.cos(heading_error)})')
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

        # ----------------- Optional smoothing based on heading error ----------------- @
        heading_threshold_rad = np.deg2rad(20)
        weight = np.clip(1.0 - (abs(heading_error) / heading_threshold_rad), 0.0, 1.0)
        speed = np.clip(speed, min_speed, max_speed) * weight
        heading = np.clip(heading * (1.0 - weight), -0.5, 0.5)
        # self.get_logger().info(f"Heading here is {heading} and error is {heading_error}")

        # Increment the goal index if very close to the current goal
        if distance_to_goal < 0.25:
            speed = 0.0
            heading = 0.0
            self.path_idx += 1
            self.get_logger().info("increment")

        return speed, heading
    
    def getPath(self, current_waypoint):
        start_map_pos = world2map(self.map, self.bot_position)
        goal_pose = map2world(self.map, current_waypoint)

        self.GoalPub(goal_pose)
                
        a_star = AStar(start_pos_vec=start_map_pos, 
                    end_pos_vec=current_waypoint, occupancy_map=self.map_array)
        
        path = a_star.find_path_smooth()

        if path == None:
            self.get_logger().info(f'A* Path failed from {start_map_pos} to {current_waypoint}')
            self.astar_fail_ctr += 1 
            os.system("ffplay -nodisp -autoexit ding_small.mpga")
            if self.astar_fail_ctr > 5:
                self.move_ttbot(speed=-0.1, heading=0.0)
                time.sleep(0.3)
                self.StopNow()
                time.sleep(0.3)
                self.bot_state =-1
            # self.DispMap(map_array=self.map_array, x1=start_map_pos[0], y1 = start_map_pos[1], 
            #                 x2=current_waypoint[0], y2 = current_waypoint[1], stopTime=1.5)
            return None
        
        self.astar_fail_ctr = 0
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map" 
        self.get_logger().info(f'Path:{path}')
        self.path_world = []
        for ij in path:
            xy_point = map2world(self.map, ij)
            self.path_world.append(xy_point)
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "map" 
            pose_stamped.pose.position.x = xy_point[0]
            pose_stamped.pose.position.y = xy_point[1]
            pose_stamped.pose.position.z = 0.0 
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

        self.get_logger().info(f'A* generated a path with {len(path)} points.')
        os.system("ffplay -nodisp -autoexit ting_small.mpga")
        return path
    
    def GoalPub(self, point):
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "map"  
        point_msg.point.x = point[0]
        point_msg.point.y = point[1]
        point_msg.point.z = 0.0 
        self.goal_pub.publish(point_msg)
    
    def ImageCentering(self, x, w):
        try:
            bbox_center = x + (w/2)
        except:
            return False
        ref_x = 730
        ref_scan = 0.5

        # Not looking straight at
        if np.abs(bbox_center - ref_x) > 100:
            K_p = -0.0008
            ang = np.clip(K_p * (bbox_center - ref_x), -0.8, 0.8)
            self.move_ttbot(speed=0.0, heading=ang)
            self.get_logger().info(f'Not looking straight at {np.abs(bbox_center - ref_x)}')
            return False
        elif np.abs(self.laserscan_x-ref_scan) > 0.2:
            K_p = 1.0
            speed = np.clip(K_p * (self.laserscan_x-ref_scan), -0.3, 0.3)
            self.get_logger().info(f"Speed : {speed} and {(self.laserscan_x-ref_scan)}")
            self.move_ttbot(speed=speed, heading=0.0)
            return False
        else:
            return True


    ###
    # Auxilary function
    ##3
    def imageDetection(self):
        """
        Detects the largest Red, Green, and Blue objects in self.frame.

        Returns:
            A tuple containing the bounding boxes (x, y, w, h) for Red, Green,
            and Blue objects respectively. Returns None for a color if no
            significant object of that color is found.
        """
        if self.frame is None:
            print("Error: Frame is None.")
            return None, None, None

        frame = self.frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red 
        lower_red1 = np.array([0, 200, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 200, 70])
        upper_red2 = np.array([179, 255, 255])

        # Green
        lower_green = np.array([35, 155, 70]) 
        upper_green = np.array([75, 255, 255])

        # Blue
        lower_blue = np.array([100, 200, 70]) 
        upper_blue = np.array([130, 255, 255])

        kernel_size = (7, 7)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        # --- Detect Colors using the Helper Function ---
        red_bbox = find_largest_contour_by_color(
            hsv_frame, lower_red1, upper_red1, lower_red2, upper_red2, morph_kernel
        )
        green_bbox = find_largest_contour_by_color(
            hsv_frame, lower_green, upper_green, kernel=morph_kernel
        )
        blue_bbox = find_largest_contour_by_color(
            hsv_frame, lower_blue, upper_blue, kernel=morph_kernel
        )

        if green_bbox and blue_bbox:
            if green_bbox[3] < blue_bbox[3]:
                green_bbox = None
            else:
                blue_bbox = None

        self.get_logger().info(f"Red = {red_bbox}, Blue is {blue_bbox}, Green = {green_bbox}. {self.colors_checklist}")
        

        if self.colors_checklist["red"]:
            red_bbox = None
        if self.colors_checklist["blue"]:
            blue_bbox = None
        if self.colors_checklist["green"]:
            green_bbox = None

        # Return the results (each will be a tuple (x,y,w,h) or None)
        return red_bbox, green_bbox, blue_bbox

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

    def StopNow(self):
        self.move_ttbot(0.0,0.0)
        self.get_logger().info('Stopping!!')

    def estimate_distance_from_bbox(self, w, h):
        """

        Args:
            w (int or float): Width of the bounding box in pixels.
            h (int or float): Height of the bounding box in pixels.

        Returns:
            float: Estimated distance to the center of the ball in meters.
                Returns None if width or height are invalid (<= 0 or None).
        """
        K_FACTOR_METER_PIXELS = 565
        # Validate inputs
        if w is None or h is None or w <= 0 or h <= 0:
            self.get_logger().warn("Warning: Invalid bounding box dimensions (w or h <= 0 or None). Cannot estimate distance.")
            self.bot_state = 1
            return None

        # Calculate the average apparent size from the bounding box dimensions
        apparent_size_pixels = (w + h) / 2.0

        if np.abs(w -h) > 20:
            self.get_logger().info("Object occlusion possible")
            apparent_size_pixels = max(w,h)

        estimated_dist_meters = K_FACTOR_METER_PIXELS / apparent_size_pixels

        return estimated_dist_meters
    
    def imageBBOX(self):
        red, green, blue = self.imageDetection()
        x = None
        y = None
        w = None
        h = None
        color = None
        if red and not self.colors_checklist["red"]:
            x, y, w, h = red
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red rect
            self.get_logger().info(f"I see red x: {x}, y: {y}, w: {w}, h: {h}")
            color = "red"
        if green and not self.colors_checklist["green"]:
            x, y, w, h = green
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Green rect
            self.get_logger().info(f"I see green x: {x}, y: {y}, w: {w}, h: {h}")
            color = "green"
        if blue and not self.colors_checklist["blue"]:
            x, y, w, h = blue
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Blue rect
            self.get_logger().info(f"I see blue x: {x}, y: {y}, w: {w}, h: {h}")
            color = "blue"
        
        return x,y,w,h,color
    
    def getPotentialEnd(self, x, w, h, dist):
        front = self.bot_heading_vec
        left = (-front[1], front[0])

        if (x + w/2) < 700:
            potential_end_point = (self.bot_position[0] + dist * front[0] + (dist / 10) * left[0], 
                                   self.bot_position[1] + dist * front[1] + (dist / 10) * left[1])
        elif (x + w/2) > 900:
            potential_end_point = (self.bot_position[0] + dist * front[0] - (dist / 10) * left[0], 
                                   self.bot_position[1] + dist * front[1] - (dist / 10) * left[1])
        else:
            potential_end_point = (self.bot_position[0] + dist * front[0], 
                                   self.bot_position[1] + dist * front[1])
            
        return potential_end_point
    
    def is_collision_free(self, pos_vec1, pos_vec2):
        line_vec = np.array(pos_vec2) - np.array(pos_vec1)
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return True

        sampling_resolution = 0.5
        num_checks = max(2, int(np.ceil(line_len / sampling_resolution)))

        for i in range(num_checks + 1):
            t = min(i / num_checks, 1.0)
            check_point = pos_vec1 + t * line_vec
            map_x = int(np.floor(check_point[0]))
            map_y = int(np.floor(check_point[1]))
            if not (0 <= map_x < self.map_dat.width -5 and 0 <= map_y < self.map_dat.height -5):
                return False

            if self.map_array[map_y, map_x] != 0:
                return False

        return True
    ##
    # Main Run function
    ##
    
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
        drive_ctr_max = 300

        while True:
            rclpy.spin_once(self, timeout_sec=0.2)
            curr_time = self.get_clock().now().nanoseconds * 1e-9
            match self.bot_state:
                case 0:
                    self.get_logger().info("Navigating...")
                    drive_ctr = 1e6
                    if all(self.waypoint_checklist.values()):
                        break
                    
                    bot_ij = world2map(self.map, self.bot_position)
                    smallest_dist = np.inf
                    for k, v in self.waypoint_checklist.items():
                        dist = np.linalg.norm(np.array(k) - np.array(bot_ij))
                        if v == False and (dist < smallest_dist):
                            current_waypoint = k
                            smallest_dist = dist
                    
                    path = self.getPath(current_waypoint=current_waypoint)

                    if path is None:
                        continue

                    self.path_idx = 0 # Reset path index for the new path
                    self.bot_state = 1
                
                case 1:
                    current_goal = self.path_world[self.path_idx]
                    speed, heading = self.path_follower(current_goal_pose=current_goal)
                    self.move_ttbot(speed=speed, heading=heading)

                    if self.path_idx >= len(self.path_world):
                        self.waypoint_checklist[current_waypoint] = True
                        self.bot_state = 2
                        self.move_ttbot(speed=0.0, heading=1.0)
                        time.sleep(0.2)
                        time_loop_start = curr_time
                        continue
                 
                    waypoint = PointStamped()
                    waypoint.header.frame_id = "map"  
                    waypoint.point.z = 0.0 
                    waypoint.point.x = current_goal[0]
                    waypoint.point.y = current_goal[1]
                    self.waypoint_pub.publish(waypoint)

                    if drive_ctr > drive_ctr_max:
                        red, green, blue = self.imageDetection()
                        drive_ctr = 0
                        if (red and not self.colors_checklist["red"]) or (blue and not self.colors_checklist["blue"]) or (green and not self.colors_checklist["green"]):
                            self.bot_state = 3
                    
                    drive_ctr += 1
                    # self.get_logger().info(f"Driving to {current_goal}")
                
                case 2:
                    self.move_ttbot(0.0, 0.8)
                    self.get_logger().info(f'{curr_time - time_loop_start}')
                    if ((curr_time - time_loop_start) > 10):
                        self.bot_state = 0
                        self.StopNow()
                        
                    self.get_logger().info(f"Looking around room.. checklist is {self.colors_checklist}")
                    red, green, blue = self.imageDetection()
                    
                    if (red and not self.colors_checklist["red"]) or (blue and not self.colors_checklist["blue"]) or (green and not self.colors_checklist["green"]):
                        # cv2.imshow("Detections", self.frame)
                        # cv2.waitKey(1)
                        self.bot_state = 3
                        # cv2.destroyAllWindows()
                
                case 3:
                    self.StopNow()
                    time.sleep(0.3)
                    x,y,w,h, color = self.imageBBOX()

                    potential_distance = self.estimate_distance_from_bbox(w=w, h=h)
                    try:
                        potential_end_point = self.getPotentialEnd(x=x, w=w, h=h, dist=potential_distance)
                    except:
                        continue
                    
                    if potential_distance < 2:
                        self.bot_state = 4
                        continue

                    if potential_distance < 4:
                        drive_ctr_max = 100
                    
                    ij = world2map(self.map, potential_end_point)

                    # self.DispMap(self.map_array, x1=ij[0], y1=ij[1])

                    self.GoalPub(potential_end_point)

                    while (ij[0] > self.map_dat.width -5 or ij[1] > self.map_dat.height -5) or (ij[0] < 0 or ij[1] < 0):
                        potential_distance -= 0.01
                        if potential_distance < 0:
                            self.move_ttbot(0.0, 1.0)
                            time.sleep(0.2)
                            break
                        potential_end_point = self.getPotentialEnd(x=x, w=w, h=h, dist=potential_distance)
                        ij = world2map(self.map, potential_end_point)
                        if (ij[0] > self.map_dat.width -5 or ij[1] > self.map_dat.height -5):
                            self.get_logger().info(f'Map out of bounds')
                            break
                        if (ij[0] < 0 or ij[1] < 0):
                            self.bot_state = 2
                            self.get_logger().info(f'Map out of bounds')
                            break
                        self.GoalPub(potential_end_point)
                    
                    while not(self.map_array[ij[1]][ij[0]] == 0):
                        potential_distance -= 0.01
                        if potential_distance < 0:
                            self.move_ttbot(0.0, 1.0)
                            time.sleep(0.2)
                            break
                        potential_end_point = self.getPotentialEnd(x=x, w=w, h=h, dist=potential_distance)
                        self.GoalPub(potential_end_point)
                        ij = world2map(self.map, potential_end_point)
                        if (ij[0] > self.map_dat.width -5 or ij[1] > self.map_dat.height -5):
                            self.get_logger().info(f'Map out of bounds')
                            break
                        if (ij[0] < 0 or ij[1] < 0):
                            self.bot_state = 2
                            self.get_logger().info(f'Map out of bounds')
                            break

                    if self.map_array[ij[1]][ij[0]] == 0:
                        path = self.getPath(current_waypoint=ij)
                        while path is None:
                            rclpy.spin_once(self, timeout_sec=0.1)
                            
                            potential_end_point = self.getPotentialEnd(x=x,w=w,h=h,dist=potential_distance)
                            self.GoalPub(potential_end_point)

                            ij = world2map(self.map, potential_end_point)
                            bot_ij = world2map(self.map, self.bot_position)

                            if self.is_collision_free(pos_vec1=ij, pos_vec2=bot_ij):
                                path = self.getPath(current_waypoint=ij)

                            potential_distance -= 0.02
                            if potential_distance < 0:
                                self.move_ttbot(0.0, 1.0)
                                time.sleep(0.2)
                                break

                        self.path_idx = 0
                        self.bot_state = 1

                case 4:
                    x,y,w,h, color = self.imageBBOX()
                    while not self.ImageCentering(x=x, w=w):              
                        rclpy.spin_once(self, timeout_sec=0.1)
                        x,y,w,h, color = self.imageBBOX()
                        if x is None or y is None or w is None or h is None:
                            self.move_ttbot(speed=0.0, heading=-0.3)
                            self.get_logger().info("Ball went out of frame")
                            continue
                            
                        if np.abs(w - h) > 15:
                            self.get_logger().info("Occlusion possible, caution!")
                    
                    self.StopNow()
                    time.sleep(0.2)
                    rclpy.spin_once(self, timeout_sec=0.2)

                    ball_point = (self.bot_position[0] + (self.laserscan_x + 0.14) * self.bot_heading_vec[0], 
                                self.bot_position[1] + (self.laserscan_x + 0.14) * self.bot_heading_vec[1])
                    
                    self.move_ttbot(speed=-0.2, heading=0.0)
                    time.sleep(0.6)
                    
                    self.colors_checklist[color] = True
                    color_point = PointStamped()
                    color_point.header.frame_id = "map"  
                    color_point.point.z = 0.0 
                    color_point.point.x = ball_point[0]
                    color_point.point.y = ball_point[1]
                    match color:
                        case "red":
                            red_ball = ball_point
                            self.red_pub.publish(color_point)
                        case "green":
                            green_ball = ball_point
                            self.green_pub.publish(color_point)
                        case "blue":
                            blue_ball = ball_point
                            self.blue_pub.publish(color_point)
                    os.system("ffplay -nodisp -autoexit ding-36029.mp3")
                    self.bot_state = 0
                    self.StopNow()

                    x1, y1 = world2map(self.map, ball_point)
                    self.map_array[y1-6:y1+6, x1-6:x1+6] = 101
                    self.DispMap(map_array=self.map_array, stopTime=1.5)
                    drive_ctr_max = 300

                case -1:
                    self.move_ttbot(-0.25, 0.01)
                    time.sleep(0.2)
                    self.get_logger().info("CRISIS REPLANING PATH")
                    current_waypoint = np.array(self.bot_position) + 1.0 * (self.away_from_wall / np.linalg.norm(self.away_from_wall))

                    self.path_follower(current_waypoint)
                    self.crisis_ctr += 1

                    if self.crisis_ctr > 5:
                        self.bot_state = 0
            
        if all(self.colors_checklist.values()):
            self.get_logger().info(f"FOUND ALL BALLS! Red at {red_ball}, Green at {green_ball} and Blue at {blue_ball}")
        else:
            self.get_logger().info(f"Pathed to all waypoints, did not find 3 ballz {self.colors_checklist}")
            self.DispMap(map_array=self.map_array,stopTime=60)


def main(args=None):
    rclpy.init(args=args)

    task2 = Task3()

    try:
        task2.run()
        # rclpy.spin(task2)
    except KeyboardInterrupt:
        task2.get_logger().error(f'An error occurred: {e}')
        task2.get_logger().error(traceback.format_exc())
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()

## User defined functions

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

        inflated_map[row_start:row_end, col_start:col_end][inflated_map[row_start:row_end, col_start:col_end] < 100] = 100


    return inflated_map

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

def find_largest_contour_by_color(hsv_image, lower_bound1, upper_bound1,
                                  lower_bound2=None, upper_bound2=None,
                                  kernel=None):
    """
    Finds the largest contour within specified HSV range(s).

    Args:
        hsv_image: The input image in HSV color space.
        lower_bound1: The lower HSV bound (NumPy array).
        upper_bound1: The upper HSV bound (NumPy array).
        lower_bound2: Optional lower HSV bound for colors like red that wrap around.
        upper_bound2: Optional upper HSV bound for colors like red that wrap around.
        kernel: Optional structuring element for morphological operations.

    Returns:
        A tuple (x, y, w, h) representing the bounding box of the largest contour,
        or None if no contours are found.
    """
    # --- Create mask(s) ---
    mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)
    mask = mask1 # Initialize mask with the first range

    if lower_bound2 is not None and upper_bound2 is not None:
        mask2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)
        mask = cv2.bitwise_or(mask1, mask2) # Combine masks using bitwise OR

    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --- Find Contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Find Largest Contour and Bounding Box ---
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100: 
             bounding_box = cv2.boundingRect(largest_contour) # (x, y, w, h)
             return bounding_box
        else:
             return None 
    else:
        return None 
    


if __name__ == '__main__':
    main()
