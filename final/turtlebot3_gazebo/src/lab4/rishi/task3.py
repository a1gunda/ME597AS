#!/usr/bin/env python3
# Rishikesh Gadre - Task 3 Final Project

# Imports
import rclpy
import cv2
import yaml
import numpy as np
import math
from queue import PriorityQueue
import cv_bridge

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image

# Task 3 Class
class Task3(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self):
        super().__init__('task3_node')

        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()  # Returns Position and Quaternion
        self.ttbot_pose = PoseStamped()  # Returns Position and Quaternion
        self.p_ind = 0  # The index of path

        # Image Processing Variablces
        self.bridge = cv_bridge.CvBridge()  # To convert ROS Image messages to OpenCV format
        self.curr_img = None  

        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', 10)

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)  # Covariance is not needed here
        self.create_subscription(Image, '/camera/image_raw', self.__camera_cbk, 10)  # Subscribe to camera image topic

        # Timer Creation for Execution
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Node rate
        self.rate = self.create_rate(20)  # Change this value due to simulation lag

        # Load/Inflate map image
        yaml_path = '/home/me597/ros2_ws/sim_ws/src/turtlebot3_gazebo/maps/map.yaml'
        pgm_path = '/home/me597/ros2_ws/sim_ws/src/turtlebot3_gazebo/maps/map.pgm'

        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)

        # Get necessary map values for correct robot/path/pose orientation
        self.res = map_metadata['resolution']
        self.origin = map_metadata['origin']

        # Inflate obstacles to add a safety margin around them
        map_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        map_img = cv2.threshold(map_img, 200, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = np.ones((5, 5), np.uint8)  # Adjust the size of the kernel for more/less inflation
        inflated_map = cv2.dilate(map_img, kernel, iterations=3)
        self.map = cv2.bitwise_not(inflated_map)
        self.get_logger().info('Map Detailed')

        # Color range for obstacle detection 
        self.lower_color = np.array([0, 0, 0]) 
    #   self.upper_color = np.array([101, 255, 24]) 
        self.upper_color = np.array([100, 255, 25]) 

        
       
    # Path Following Functions --------------------------------
    def timer_cb(self):
        """! Path Following Code
        @param
        @return Twist Messages
        """

        #self.get_logger().info('Task3 Node is alive!')

        # Check if obstacle is in front of robot
        if self.curr_img is not None:
            if self.find_obs(self.curr_img):

                self.get_logger().info('Obstacle detected! Stopping robot.')
                self.cmd_vel_pub.publish(Twist()) 

                return

        # Follow the path to intermediate waypoints
        if self.p_ind < len(self.path.poses):
            waypoint = self.path.poses[self.p_ind].pose.position
            goal_x = waypoint.x
            goal_y = waypoint.y
            distance, angle_diff = self.calc_vals(goal_x, goal_y)

            twist_msg = Twist()

            # Stop the robot if it reaches the final goal point
            final_goal_distance = self.calc_distance_to_goal()
            if self.p_ind == len(self.path.poses) - 1 or final_goal_distance < 0.05:
                self.get_logger().info('Reached final goal point! Stopping and orienting correctly.')
                self.cmd_vel_pub.publish(Twist())  # Stop movement

                # Caluclates the final heading angle
                final_heading = self.heading_calc(self.goal_pose.pose.orientation)
                heading_diff = final_heading - self.heading_calc(self.ttbot_pose.pose.orientation)
                heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi

                # Align to final heading angle
                if abs(heading_diff) > math.radians(5):
                    twist_msg.angular.z = 0.15 * heading_diff
                    self.cmd_vel_pub.publish(twist_msg)
                else:
                    self.cmd_vel_pub.publish(Twist())  # Stop any movement
                return

            # Path following

            # If there is a significant heading difference robot orients itself
            if abs(angle_diff) > math.radians(20):
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.15 * angle_diff
            else:
                # Normal movement
                twist_msg.linear.x = 0.18
                twist_msg.angular.z = 0.2 * angle_diff

            # Prevent the velocities from getting too big
            if twist_msg.angular.z > 0.2:
                twist_msg.angular.z = 0.2
            elif twist_msg.angular.z < -0.2:
                twist_msg.angular.z = -0.2

            self.cmd_vel_pub.publish(twist_msg)

            # If the robot is close enough to the current waypoint, move to the next one
            if distance < 0.4:
                self.get_logger().info(f'Reached waypoint: ({goal_x:.2f}, {goal_y:.2f})')
                self.p_ind += 1
               
    def calc_vals(self, goal_x, goal_y):
        """! Calculates the distance and angle to a goal point.
        @param goal_x, goal_y  Coordinates of the goal.
        @return Tuple containing distance and angle difference.
        """
        current_x = self.ttbot_pose.pose.position.x
        current_y = self.ttbot_pose.pose.position.y
        current_orientation = self.ttbot_pose.pose.orientation
        current_heading = self.heading_calc(current_orientation)

        x_diff = goal_x - current_x
        y_diff = goal_y - current_y

        dist = np.sqrt(x_diff**2 + y_diff**2)
        goal_heading = math.atan2(y_diff, x_diff)
       
        angle_diff = goal_heading - current_heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
       
        return dist, angle_diff

    def calc_distance_to_goal(self):
        """! Calculates the distance between the current position and the final goal position.
        @return Distance to the final goal.
        """
        goal_x = self.path.poses[-1].pose.position.x
        goal_y = self.path.poses[-1].pose.position.y
        current_x = self.ttbot_pose.pose.position.x
        current_y = self.ttbot_pose.pose.position.y

        x_diff = goal_x - current_x
        y_diff = goal_y - current_y

        return np.sqrt(x_diff**2 + y_diff**2)

    # Conversion Functions --------------------------------

    def convert_coordinates(self, x, y, grid=True):
        """! Converts coordinates between grid and world frames.
        @param x, y         Coordinates to be converted.
        @param to_grid      Boolean flag to determine conversion direction
        @return Coordinate tuple (grid or pose positioning).
        """
        if grid:
            grid_x = int((x - self.origin[0]) / self.res)
            grid_y = self.map.shape[0] - int((y - self.origin[1]) / self.res)  
            return (grid_x, grid_y)
        else:
            world_x = x * self.res + self.origin[0]  
            world_y = (self.map.shape[0] - y) * self.res + self.origin[1]  
            return (world_x, world_y)

    def heading_calc(self, q):
        """! Rotates the Quaternion around to calculate the heading of the robot
        @param q
        @return Heading
        """
        val_1 = 2.0 * (q.w * q.z + q.x * q.y)
        val_2 = 2.0 * (q.y * q.y + q.z * q.z)

        theta = math.atan2(val_1, 1.0 - val_2)

        return theta

    # A Star Algorithm Functions --------------------------------

    def heuristic(self, point, goal):
        """! Calculates the heuristic value for the given node
        @param point, goal
        @return heuristic value
        """
        return np.sqrt((point[0] - goal[0]) ** 2 + (point[1] - goal[1]) ** 2)

    def reconstruct_path(self, prev_pt, current_node):
        """! Creates the Path
        @param prev_pt, current_node
        @return Path
        """
        # Reconstructs the path
        path = [current_node]
        while current_node in prev_pt:
            current_node = prev_pt[current_node]
            path.append(current_node)
        path.reverse()
        return path

    def get_neighbors(self, node):
        """! Gets the valid neighbors for a given node.
        @param node Coordinates of the current node.
        @return List of valid neighbor nodes.
        """
        neighbors = []
        grid_next = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for direction in grid_next:
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if 0 <= neighbor[0] < self.map.shape[1] and 0 <= neighbor[1] < self.map.shape[0] and self.map[neighbor[1], neighbor[0]] != 0:
                neighbors.append(neighbor)

        return neighbors

    def a_star_path_planner(self, start, goal):
        """! A* path planner.
        @param  start    Tuple containing the start coordinates.
        @param  goal     Tuple containing the goal coordinates.
        @return path     Path object containing the sequence of waypoints of the created path.
        """
        open_list = PriorityQueue()  
        came_from = {}  
        g_score = {start: 0}  
        f_score = {start: self.heuristic(start, goal)}  

       
        open_list.put((f_score[start], start))
        while not open_list.empty():
           
            current_node = open_list.get()[1]

            # If we have reached the goal, return the path  
            if current_node == goal:
                self.get_logger().info('A* Path!')
                return self.reconstruct_path(came_from, current_node)

            # Checks to see if neighboring nodes are occupied
            for neighbor in self.get_neighbors(current_node):
                # Tentative G score calculation
                tentative_g_score = g_score[current_node] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                   
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    open_list.put((f_score[neighbor], neighbor))

       
        self.get_logger().info('No path found')
        return None

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.get_logger().info(
            'Goal pose received: ({:.4f}, {:.4f})'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

        # Publish Goal as Marker
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = self.goal_pose.pose
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.goal_marker_pub.publish(marker)

        # A* Path Planning
        start = self.convert_coordinates(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y, grid=True)
        goal = self.convert_coordinates(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y, grid=True)
        path = self.a_star_path_planner(start, goal)

        # Path conversion and publishing
        if path:
            world_path = [self.convert_coordinates(x, y, grid=False) for x, y in path]

            path_msg = Path()
            path_msg.header.frame_id = 'map'
            path_msg.header.stamp = self.get_clock().now().to_msg()
           
            for x, y in world_path:
                pose = PoseStamped()
                pose.pose.position.x = x
                pose.pose.position.y = y
                path_msg.poses.append(pose)

            # Publish the planned path
            self.path_pub.publish(path_msg)
            self.path = path_msg
            self.p_ind = 0

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'Current turtlebot pose updated: ({:.4f}, {:.4f})'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

           
    # Dynamic Obstacle Detection Functions --------------------------------
    def __camera_cbk(self, msg):
        """! Callback to get the camera image.
        @param msg Image message from the camera.
        @return None.
        """
        try:
            self.curr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def find_obs(self, image):
        """! Detects if an obstacle with a specific color is in view.
        @param image Image from the camera.
        @return Boolean indicating if an obstacle is detected.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
        mask = cv2.inRange(hsv_image, self.lower_color, self.upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 285000:  # Area needed to stop when obstacle is visible
                # Get bounding box
                self.get_logger().info(f'area: {area:.2f}')
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the bounding box matches the expected size of the trash can
                if 0.1 < w / h < 2.0: 
                    return True
        return False
def main(args=None):
    rclpy.init(args=args)

    task3 = Task3()

    try:
        rclpy.spin(task3)
    except KeyboardInterrupt:
        pass
    finally:
        task3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
