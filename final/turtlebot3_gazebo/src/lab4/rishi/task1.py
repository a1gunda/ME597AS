#!/usr/bin/env python3
# Rishikesh Gadre - Task 1 Final Project

# Imports
import rclpy
import numpy as np
import time
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from sklearn.cluster import DBSCAN
from queue import PriorityQueue


# Task 1 Class
class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')  

        # Executable Function
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.frontier_marker_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.cluster_marker_pub = self.create_publisher(MarkerArray, '/cluster_markers', 10)
        self.target_marker_pub = self.create_publisher(Marker, '/current_target_marker', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # Map Path Planning/Movement variables
        self.map_data = None  # Store occupancy grid data
        self.curr_pos = (0.0, 0.0)
        self.curr_head = 0.0  # Radians
        self.cluster_points = []
        self.start = time.time()
        self.spinning = True
        self.curr_goal = None
        self.path = None  
        self.p_ind = 0  
        

    # Call Back Functions --------------------------------
    def map_cb(self, msg):
        """! Updates the Map When there is no target/path
        @param
        @return
        """
        if self.curr_goal is None and self.path is None:
            self.map_data = msg
            self.find_frontiers()

    def odom_cb(self, msg):
        """! Extracts the odometery data of the robot
        @param
        @return Twist Messages
        """
        self.curr_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        self.curr_head = self.heading_calc(orientation_q)

        # If following a path, continue to the next waypoint
        if self.path is not None:
            self.follow_path()

    # Frontier and Cluster Calculation Functions --------------------------------
    def find_frontiers(self):
        """! Checks the occupancy grid for unsearched cells (froniters)
        @param 
        @return Frontiers
        """
        # If no map data nothing is published
        if self.map_data is None:
            return

        w = self.map_data.info.width
        h = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        # Convert map data to find frontiers through the occupancy grid
        grid = np.array(self.map_data.data).reshape((h, w))

        frontiers = []

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if grid[y, x] == 0:  # Free space
                    # Check if the cell is adjacent to unknown space (-1)
                    neighbors = [grid[y + 1, x], grid[y - 1, x],
                                 grid[y, x + 1], grid[y, x - 1]]
                    
                    # Check if th cell is a frontier 
                    if -1 in neighbors:
                        frontier_x = origin_x + (x * resolution)
                        frontier_y = origin_y + (y * resolution)

                        # Add frontier to the list
                        frontiers.append((frontier_x, frontier_y))

        # Publish frontiers as markers
        self.publish_frontiers(frontiers)

        # Cluster Calculations
        self.cluster_calcs(frontiers)

    def cluster_calcs(self, frontiers):

        # If no frontier data nothing is published
        if not frontiers:
            return

        # Convert frontiers to data
        frontier_points = np.array(frontiers)

        # Use DBSCAN to find mean of frontier points to create a clutser point
        if len(frontier_points) < 10:
            eps = 0.3
        else:
            eps = 0.5

        clustering = DBSCAN(eps=eps, min_samples=2).fit(frontier_points)

        cluster_centers = []
        labels = clustering.labels_
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points

            # Get all points in the current cluster
            cluster_points = frontier_points[labels == label]
            # Calculate the cluster center as the mean of the points
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append((cluster_center[0], cluster_center[1]))

        # Publish cluster centers as markers
        self.publish_clusters(cluster_centers)
        self.cluster_points = cluster_centers  # Store the latest cluster centers

    # Target Cluster Selection --------------------------------
    def select_next_target(self):
        
        # Recalculate frontiers and clusters before selecting a new target
        self.find_frontiers()

        if not self.cluster_points:
            return None

        best_cluster = None
        best_score = float('inf')

        # Updates Map Parameters
        self.map_resolution = self.map_data.info.resolution
        self.map_width = self.map_data.info.width
        self.map_height = self.map_data.info.height
        self.map_origin_x = round(self.map_data.info.origin.position.x, 1)
        self.map_origin_y = round(self.map_data.info.origin.position.y, 1)

        # Initialize self.origin
        self.origin = (self.map_origin_x, self.map_origin_y)
        self.map_shape = (self.map_height, self.map_width)

        # Scores clusters based on distance, frontier size, and how close they are to walls walls
        for cluster in self.cluster_points:
            # Calculate the distance from the current position to the cluster
            dist = self.heuristic(self.curr_pos, cluster)

            # Estimate the frontier size
            frontier_size = 0
            for c in self.cluster_points:
                if self.heuristic(cluster, c) < 1.0:
                    frontier_size += 1

            # Check if the cluster is on a wall, if so it skips it as an option
            cluster_gridx, cluster_gridy = self.convert_coordinates(cluster[0], cluster[1], grid=True)

            if self.map_data.data[cluster_gridy * self.map_width + cluster_gridx] == 100:
                continue

            # Check the value in the map data at the specified grid coordinate
            wall_bias = 9.0 if self.map_data.data[cluster_gridy * self.map_width + cluster_gridx] >= 80 else 0.0

            head = math.atan2(cluster[1] - self.curr_pos[1], cluster[0] - self.curr_pos[0])
            head_diff = abs(math.atan2(math.sin(head - self.curr_head), math.cos(head - self.curr_head)))

            # Generates score to minimize distance, maximize frontier size, and avoid walls
            score = dist - frontier_size + 2.0 * head_diff + wall_bias

            if score < best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster:

            # Artificial cluster creation in forward direction if no viable cluster
            if best_score > 10.0:
                forward_x = self.curr_pos[0] + math.cos(self.curr_head) * 0.5
                forward_y = self.curr_pos[1] + math.sin(self.curr_head) * 0.5
                best_cluster = (forward_x, forward_y)
                self.get_logger().info(f'Artificial cluster created at: {best_cluster}')

            self.curr_goal = best_cluster
            self.publish_target(best_cluster)
            self.get_logger().info(f'Next target selected: {best_cluster}')
            # Plan a path to the selected target
            if self.path is None:
                self.plan_path_to_target()

    # Frontier, Cluster, and Target Visualization --------------------------------
    def publish_frontiers(self, frontiers):
        """! Publishes the frontier points
        @param
        @return 
        """
        marker_array = MarkerArray()
        for idx, (fx, fy) in enumerate(frontiers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "frontiers"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = fx
            marker.pose.position.y = fy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05  
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0  
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.frontier_marker_pub.publish(marker_array)

    def publish_clusters(self, cluster_centers):
        """! Publishes the cluster points
        @param
        @return 
        """
        marker_array = MarkerArray()
        for idx, (cx, cy) in enumerate(cluster_centers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "clusters"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0  
            marker.color.r = 0.0
            marker.color.g = 0.5
            marker.color.b = 1.0  
            marker_array.markers.append(marker)

        self.get_logger().info('Cluster Points Published')
        self.cluster_marker_pub.publish(marker_array)

    def publish_target(self, target):
        """! Publishes the current target marker
        @param
        @return 
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target[0]
        marker.pose.position.y = target[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4  
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.a = 1.0  
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0 

        self.get_logger().info('Goal Published!')
        self.target_marker_pub.publish(marker)

    # A Star Algorithm --------------------------------
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
        DOF = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for direction in DOF:
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if 0 <= neighbor[0] < self.map_width and 0 <= neighbor[1] < self.map_height:
                if self.check_neighbor(neighbor):
                    neighbors.append(neighbor)

        return neighbors

    def check_neighbor(self, node):
        """ Checks if the given node is not too close to obstacles).
        @param node Coordinates of the node to check.
        @return Boolean indicating if the node is safe.
        """
        buffer_distance = 3  
        x, y = node

        for dx in range(-buffer_distance, buffer_distance + 1):
            for dy in range(-buffer_distance, buffer_distance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if int(self.map_data.data[ny * self.map_width + nx]) >= 50:
                        # Neighbor is too close to an obstacle
                        return False
        return True

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
                self.get_logger().info('A* path found')
                return self.reconstruct_path(came_from, current_node)

            # Checks to see if neighboring nodes are occupied
            for neighbor in self.get_neighbors(current_node):
                tentative_g_score = g_score[current_node] + 1  

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    open_list.put((f_score[neighbor], neighbor))

    
        self.get_logger().info('No path found')

        # self.problem = self.current_target
        # if self.current_target in self.cluster_points:
        #     self.cluster_points.remove(self.current_target)  # Remove the current target
        # self.current_target = None  # Reset the current target
        # self.move_forward_until_new_target()

        return None

    # Path Following Implementation --------------------------------
    def plan_path_to_target(self):
        """! Creates a path to the target
        @param  
        @return None.
        """
        if self.curr_goal is None:
            return

        # Convert current position and target position to grid coordinates
        start = self.convert_coordinates(self.curr_pos[0], self.curr_pos[1], grid=True)
        goal = self.convert_coordinates(self.curr_goal[0], self.curr_goal[1], grid=True)

        # Plan path using A*
        path = self.a_star_path_planner(start, goal)

        # Publish the path if found
        if path is not None:

            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for point in path:
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = self.get_clock().now().to_msg()
                world_point = self.convert_coordinates(point[0], point[1], grid=False)
                pose.pose.position.x = world_point[0]
                pose.pose.position.y = world_point[1]
                pose.pose.position.z = 0.0
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)
            self.get_logger().info('Path to target published')
            self.path = path  
            self.p_ind = 0  

    def follow_path(self):
        """! Path Following Code
        @param 
        @return Twist Messages
        """
        if self.path is None or self.p_ind >= len(self.path):
            return

        # Get the current target point
        target_waypoint = self.path[self.p_ind]
        target_conv = self.convert_coordinates(target_waypoint[0], target_waypoint[1], grid=False)

        # Calculate distance to the target waypoint
        dist = self.heuristic(self.curr_pos, target_conv)

        if dist < 0.1: 
            self.p_ind += 1
            if self.p_ind >= len(self.path):
                # Clears path so new path can be created
                self.get_logger().info('Reached the target')
                self.path = None  

                # Remove the current target from the cluster points
                if self.curr_goal in self.cluster_points:
                    self.cluster_points.remove(self.curr_goal)
                    self.get_logger().info('Current Target Removed')

                self.curr_goal = None  
                self.find_frontiers()

                # Stops Movement
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                return

        # Calculate the desired heading to the target waypoint
        head = math.atan2(target_conv[1] - self.curr_pos[1], target_conv[0] - self.curr_pos[0])
        head_diff = head - self.curr_head

        # Normalize heading error 
        head_diff = math.atan2(math.sin(head_diff), math.cos(head_diff))

        # P Controller for Movement
        lin_speed = 0.18
        ang_speed = 0.4 * head_diff

        twist = Twist()
        twist.linear.x = lin_speed if abs(head_diff) < 0.5 else 0.0
        twist.angular.z = ang_speed
        self.cmd_vel_pub.publish(twist)

     # Conversion Functions --------------------------------
    def convert_coordinates(self, x, y, grid=True):
        """! Converts coordinates between grid and world frames.
        @param x, y         Coordinates to be converted.
        @param to_grid      Boolean flag to determine conversion direction
        @return Coordinate tuple (grid or pose positioning).
        """
        if grid:
            x = int((x - self.origin[0]) / self.map_resolution)
            y = int((y - self.origin[1]) / self.map_resolution)
        else:
            x = (x * self.map_resolution) + self.origin[0]
            y = (y * self.map_resolution) + self.origin[1]

        return (x, y)
    
    def heading_calc(self, q):
        """! Rotates the Quaternion around to calculate the heading of the robot
        @param q
        @return Heading
        """
        val_1 = 2.0 * (q.w * q.z + q.x * q.y)
        val_2 = 2.0 * (q.y * q.y + q.z * q.z)

        theta = math.atan2(val_1, 1.0 - val_2)

        return theta

    # Executable Function --------------------------------
    def timer_cb(self):
        """! Main Executable to run Autonomous Mapping
        @param 
        @return 
        """
        # Spin the robot to see cluster poiints
        if self.spinning:

            if self.map_data is None:
                self.get_logger().info('Map data not received')
                return
            twist = Twist()
            twist.angular.z = 0.5  
            self.cmd_vel_pub.publish(twist)

            # Tuned for Lab Computer -> 5-15 fps
            if time.time() - self.start >= 35:
                self.spinning = False
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)  
        else:
            self.select_next_target()  

def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
