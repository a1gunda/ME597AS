#!/usr/bin/env python3
# ======================================================================
# imports
# ======================================================================
import os
import PIL
import cv2
import yaml
import rclpy
import heapq
import math
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from copy import copy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

# ======================================================================
# map + a* support classes
# ======================================================================

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits, self.map_cv2 = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        map_path = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'maps')
        f = open(os.path.join(map_path, map_name + '.yaml'), 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        map_name = map_df.image[0]
        im = PIL.Image.open(os.path.join(map_path, map_name))
        im = PIL.ImageOps.grayscale(im)
        map_image = cv2.imread(os.path.join(map_path, map_name), cv2.IMREAD_GRAYSCALE)
        map = cv2.threshold(map_image, 200, 255, cv2.THRESH_BINARY)[1]
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax], map

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array

class TreeNode():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)

class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}

    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True

class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.dist = {name: np.inf for name in in_tree.g}
        self.h = {}

        end = tuple(map(int, in_tree.end.split(',')))
        for name in in_tree.g:
            start = tuple(map(int, name.split(',')))
            self.h[name] = np.hypot(end[0] - start[0], end[1] - start[1])

        self.via = {name: None for name in in_tree.g}
        self.open_set = []  # priority queue
        self._counter = 0   # manual tie-breaker counter

    def __get_f_score(self, node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        heapq.heappush(self.open_set, (self.__get_f_score(sn), self._counter, sn))
        self._counter += 1

        visited = set()

        while self.open_set:
            _, __, u = heapq.heappop(self.open_set)

            if u.name in visited:
                continue
            visited.add(u.name)

            if u.name == en.name:
                break

            for c, w in zip(u.children, u.weight):
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name
                    heapq.heappush(self.open_set, (self.__get_f_score(c), self._counter, c))
                    self._counter += 1

        return self.via

    def reconstruct_path(self, sn, en):
        start_key, end_key = sn.name, en.name
        if self.dist[end_key] == np.inf:
            return [], np.inf  # no path found

        path = [end_key]
        while path[-1] != start_key:
            prev = self.via[path[-1]]
            if prev is None:
                return [], np.inf  # path broken
            path.append(prev)
        path.reverse()
        return path, self.dist[end_key]

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = TreeNode('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array

# ======================================================================
# task 3 node
# ======================================================================

class Task3(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self, node_name='task3_node'):
        super().__init__(node_name)

        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        self.follow_idx = 0
        self.goal_pose_given = False
        self.ttbot_pose_given = False
        self.replan = False
        self.plan_computed = False

        self.wp_reached_thresh = 0.25

        self.clearance_radius_cells = 4
        self.narrow_proximity_thresh = 0.15
        self.narrow_lin = 0.15

        # pid state (angular)
        self.PID_ang_int = 0.0
        self.PID_ang_prev = 0.0
        self.last_time = self.get_clock().now().nanoseconds * 1e-9

        # States: 'SEARCH', 'PLAN_TO_GOAL', 'APPROACH_BALL', 'LOCALIZED_ALL', 'RETREAT'
        self.task3_state = 'SEARCH'
        self.retreat_start_time = None # track retreat duration
        self.plan_fail_count = 0

        # Storage for localized balls: {color: [x, y], ...}
        self.found_balls = {} 
        self.target_colors = ['red', 'blue', 'green']
        self.current_target_color_idx = 0

        # --- Color HSV Ranges ---
        self.COLOR_RANGES = {
            # Dual range for Red
            'red': [
                (np.array([0, 80, 80]), np.array([10, 255, 255])),
                # (np.array([170, 80, 80]), np.array([180, 255, 255]))
            ],
            # Single range for Green
            'green': [
                (np.array([40, 80, 80]), np.array([80, 255, 255]))
            ],
            # Single range for Blue
            'blue': [
                (np.array([100, 80, 80]), np.array([140, 255, 255]))
            ]
        }

        # Vision tracking parameters (copied from RedBallTracker)
        self.trgt_area = 1e4
        self.dband = 4e2
        self.max_lin = 0.4
        self.max_ang = 0.6
        self.Kp_lin = 1.2e-4
        self.Kd_lin = 1.8e-4
        self.Kp_ang = 4.2e-4
        self.Kd_ang = 1.2e-4
        self.filt_d_area = 0.0
        self.alpha = 0.25  # Low-pass filter smoothing factor (copied from original RedBallTracker)

        # Vision memory
        self.bridge = CvBridge()
        self.cv_img = None # To hold the latest camera image
        self.filt_d_area = 0.0
        self.prev_area_err = 0
        self.prev_cntr_err = 0
        self.prev_time_vision = self.get_clock().now()

        # Visualization
        cv2.namedWindow('task_3_vision', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('task_3_vision', 640, 400)

        # ... (existing Task 3 variables) ...
        self.task3_state = 'SEARCH' 

        # Hardcoded waypoints for house coverage (in map coordinates: x, y)
        # IMPORTANT: Adjust these coordinates based on the actual map limits 
        # and known open areas to ensure thorough coverage.
        self.exploration_waypoints = [
            # Waypoint 4: Top-right room
            (7.98874, -5.48031),
            # Waypoint 4: Top-left area
            (8.5962, 2.8861),
            # Waypoint 4: Mid-left room
            (3.12934, 0.923115),
            # Waypoint 1: Mid-left area
            (0.7810, 2.6347),
            # Waypoint 2: Bottom-left area
            (-4.0107, 3.0100),
            # Waypoint 3: Bottom-right room
            (-4.1421, -3.7268),
        ]
        self.current_wp_idx = 0
        self.initial_search_complete = False # Flag to stop searching when all WPs are reached

        # subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(LaserScan, '/scan', self.__scan_cbk, 10)
        self.create_subscription(Image, '/camera/image_raw', self.__camera_cbk, 10)

        # publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time', 10)
        self.inf_map_pub = self.create_publisher(OccupancyGrid, '/inflated_map', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dynamic_obstacle_markers', 10)

        # map generation
        self.map_name = 'sync_classroom_map'
        self.mp_dyn = MapProcessor(self.map_name)
        self.kr_dyn = 4
        self.mp_dyn.inflate_map(self.mp_dyn.rect_kernel(self.kr_dyn, 1), True)

        self.mp = MapProcessor(self.map_name)
        self.kr = 11
        self.mp.inflate_map(self.mp.rect_kernel(self.kr, 1), True)
        self.mp.get_graph_from_map()

        self.res = self.mp.map.map_df.resolution[0]
        self.origin = self.mp.map.map_df.origin[0]
        self.shape = self.mp.map.image_array.shape[0]

        self.graph_built = True
        self.map_ready = True

        h, w = self.mp.map.image_array.shape
        self.get_logger().info(
            f'Local map loaded: {w}x{h}, nodes={len(self.mp.map_graph.g)}'
        )

        self.static_inf_map = np.copy(self.mp.inf_map_img_array)

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------

    def __goal_pose_cbk(self, data):
        # We accept RViz goals, but only if we are NOT actively searching.
        if self.task3_state == 'SEARCH' and not self.initial_search_complete:
            self.get_logger().info('Ignoring manual goal: Robot is executing search plan.')
            return

        self.goal_pose = data
        self.goal_pose.header.frame_id = 'map'
        self.goal_pose_given = True
        
        # Reset map processors and planning flags as before
        self.mp_dyn = MapProcessor(self.map_name)
        self.mp_dyn.inflate_map(
            self.mp_dyn.rect_kernel(self.kr_dyn, 1),
            True
        )

        self.mp = MapProcessor(self.map_name)
        self.mp.inflate_map(self.mp.rect_kernel(self.kr, 1), True)

        self.graph_built = False
        self.replan = True
        self.plan_computed = False

        self.get_logger().info(
            f'Manual goal set: {self.goal_pose.pose.position.x:.4f}, '
            f'{self.goal_pose.pose.position.y:.4f}'
        )

    def __ttbot_pose_cbk(self, data):
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose
        self.ttbot_pose.header.frame_id = 'map'
        self.ttbot_pose_given = True

        self.get_logger().info(
            f'ttbot_pose: {self.ttbot_pose.pose.position.x:.4f}, '
            f'{self.ttbot_pose.pose.position.y:.4f}'
        )

    def __scan_cbk(self, msg):
        # process lidar, update maps, trigger replanning
        if not self.ttbot_pose_given or not self.map_ready:
            return

        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = self.calc_heading(self.ttbot_pose.pose.orientation)

        ranges = np.array(msg.ranges)
        # Only consider obstacles within 1.0m and beyond the minimum range
        valid_idx = np.where((ranges < 1.0) & (ranges > 0.05))[0]

        map_changed = False

        # dimensions of the planner map
        H, W = self.mp.inf_map_img_array.shape

        kr_dyn = self.kr_dyn // 2
        kr_inf = self.kr // 5

        for i in valid_idx:
            r = ranges[i]
            theta = msg.angle_min + i * msg.angle_increment

            ox = rx + r * math.cos(yaw + theta)
            oy = ry + r * math.sin(yaw + theta)

            ix, iy = self.convert_coordinates(ox, oy, grid=True)

            # skip if out of bounds
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            
            # skip if already occupied in dynamic map (optimization)
            if self.mp_dyn.inf_map_img_array[iy, ix] != 0:
                continue

            map_changed = True
            clamp = lambda v, lo, hi: max(lo, min(hi, v))

            # dynamic map inflation
            y0 = clamp(iy - kr_dyn, 0, H)
            y1 = clamp(iy + kr_dyn + 1, 0, H)
            x0 = clamp(ix - kr_dyn, 0, W)
            x1 = clamp(ix + kr_dyn + 1, 0, W)

            self.mp_dyn.inf_map_img_array[y0:y1, x0:x1] = 100

            # planner map inflation (used by A*)
            y0 = clamp(iy - kr_inf, 0, H)
            y1 = clamp(iy + kr_inf + 1, 0, H)
            x0 = clamp(ix - kr_inf, 0, W)
            x1 = clamp(ix + kr_inf + 1, 0, W)

            self.mp.inf_map_img_array[y0:y1, x0:x1] = 100

        # check upcoming path segment for collisions (only runs if map_changed is true)
        if map_changed:
            # NOTE: Your 'collision' variable is equivalent to 'path_collision'
            path_collision = False 
            
            # 1. Check current A* path for collisions (Task 2 logic)
            if self.path.poses:
                k0 = self.follow_idx
                k1 = min(len(self.path.poses), k0 + 20) # Check 20 nodes ahead

                for k in range(k0, k1):
                    wp = self.path.poses[k].pose.position
                    wx, wy = self.convert_coordinates(wp.x, wp.y, grid=True)

                    if 0 <= wx < W and 0 <= wy < H:
                        if self.mp.inf_map_img_array[wy, wx] > 0:
                            path_collision = True
                            break

            # 2. Check a small zone right in front of the robot for immediate collision
            rx_cell, ry_cell = self.convert_coordinates(rx, ry, grid=True)
            # Check for high inflation value in a small radius around the robot 
            immediate_collision = self._proximity_inflated(rx_cell, ry_cell, 2) > 0 # Assumes _proximity_inflated exists

            # --- AVOIDANCE LOGIC BLOCK ---
            if path_collision or immediate_collision:
                self.get_logger().warn('Path or Local Zone Blocked by Dynamic Obstacle! Replanning/Stopping...')
                self.move_ttbot(0.0, 0.0)
                
                # Always clear PID state on hard stop
                self.PID_ang_int = 0.0
                self.PID_ang_prev = 0.0

                if self.task3_state == 'APPROACH_BALL':
                    # Ball approach is interrupted by a close obstacle. Force retreat.
                    self.get_logger().warn('Obstacle hit during ball approach. Starting retreat.')
                    cv2.destroyAllWindows()
                    
                    self.path = Path()            
                    self.plan_computed = False    
                    self.replan = True            
                    
                    self.retreat_start_time = self.get_clock().now().nanoseconds / 1e9
                    self.task3_state = 'RETREAT'
                    self.move_ttbot(-1.0, 0.0)

                # NOTE: We only need to replan if we are in SEARCH mode and either 
                # path_collision or immediate_collision is true.
                elif self.task3_state == 'SEARCH': 
                    self.get_logger().warn('Path blocked in SEARCH mode. Forcing replan.')
                    self.path = Path()
                    self.replan = True
                    self.graph_built = False
                    self.plan_computed = False

    def __camera_cbk(self, msg):
        try:
            self.cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------

    def publish_inflated_map(self):
        if self.mp.inf_map_img_array is None:
            return

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        meta = MapMetaData()
        meta.resolution = float(self.res)
        meta.width = self.mp.inf_map_img_array.shape[1]
        meta.height = self.mp.inf_map_img_array.shape[0]
        meta.origin.position.x = float(self.origin[0])
        meta.origin.position.y = float(self.origin[1])
        msg.info = meta

        arr = np.flipud(self.mp.inf_map_img_array)
        flat = np.where(arr.flatten() > 0, 100, 0).astype(np.int8)

        msg.data = flat.tolist()
        self.inf_map_pub.publish(msg)

    def publish_obstacle_markers(self):
        pass

    # ------------------------------------------------------------------
    # object detection
    # ------------------------------------------------------------------

    def detect_ball(self, color):
        if self.cv_img is None:
            return None, None, None # contour, area, center

        img_hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        masks = []

        # Apply all HSV ranges for the given color
        for (lower, upper) in self.COLOR_RANGES[color]:
            masks.append(cv2.inRange(img_hsv, lower, upper))
        
        # Combine masks using bitwise OR
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)

        # Clean mask
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None, None

        # Largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Simple moment calculation for center (alternative to bounding box)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return c, area, (cx, cy)
        
        return None, None, None
    
    # ------------------------------------------------------------------
    # localization
    # ------------------------------------------------------------------

    def estimate_ball_distance(self, area):
        # This is a model-specific calibration.
        # Area is inversely proportional to the square of the distance (approx)
        # Calibrate this with your robot setup!
        # Example model: dist = K / sqrt(area)
        # A simple linear fit might also work if the area range is small.
        # For now, let's use a simple estimate where 1e4 area ~ 0.5m
        if area > 0:
            # Based on trgt_area = 1e4, let's assume K=5000 is a good starting point.
            # This gives 0.5m distance at 10000 area.
            return 5000 / math.sqrt(area) 
        return 1.0 # default distance

    def localize_ball(self, cx):
        # This runs when the robot is centered and stopped (i.e., localization phase)
        
        # 1. Estimate angle to ball
        H, W, _ = self.cv_img.shape
        fx = W / 2 # image center x
        
        # FOV calibration: Find the horizontal field of view (HFOV) of your camera.
        # Typical TurtleBot3 HFOV is ~ 1.047 radians (60 degrees) or more.
        HFOV = 1.047 # 60 degrees in radians (Adjust this based on your camera)
        
        # Calculate the angle offset (radians/pixel * pixel offset)
        rad_per_pixel = HFOV / W
        pixel_offset = fx - cx 
        
        # Angle in robot frame (positive is left, negative is right)
        angle_robot_frame = pixel_offset * rad_per_pixel 

        # 2. Estimate distance (using the linear PD tracking memory)
        # The current tracking area is self.trgt_area - self.prev_area_err
        current_area = self.trgt_area - self.prev_area_err 
        r = self.estimate_ball_distance(current_area)

        # 3. Transform to World Frame
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = self.calc_heading(self.ttbot_pose.pose.orientation)

        # Total angle in world frame
        angle_world_frame = self.normalize(yaw + angle_robot_frame)

        # World coordinates (using standard polar to Cartesian conversion)
        ox = rx + r * math.cos(angle_world_frame)
        oy = ry + r * math.sin(angle_world_frame)

        return ox, oy

    # ------------------------------------------------------------------
    # ball approach and tracking
    # ------------------------------------------------------------------

    def approach_and_track(self, color):
        c, area, center = self.detect_ball(color)
        lin_vel, ang_vel = 0.0, 0.0
        
        # Update time step for PD controller
        now = self.get_clock().now()
        dt = (now - self.prev_time_vision).nanoseconds / 1e9
        self.prev_time_vision = now

        if c is None:
            self.get_logger().info(f'No {color} ball detected, searching again...')
            self.publish_inflated_map() # Ensure map updates happen
            cv2.destroyAllWindows()
            
            # re-engage navigation after losing the ball
            self.path = Path()            # Clear old path (which is now irrelevant)
            self.plan_computed = False    # Indicate the current path is invalid
            self.replan = True            # Force A* to calculate a new path
            self.task3_state = 'SEARCH'
            return
        
        # Center calculation
        cx, cy = center
        fx = self.cv_img.shape[1] / 2
        
        # Visualization (re-draw visualization logic here)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(self.cv_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(self.cv_img, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        cv2.imshow('task_3_vision', self.cv_img)
        cv2.waitKey(1)

        # linear PD control (distance)
        area_err = self.trgt_area - area
        d_area = (area_err - self.prev_area_err) / dt if dt > 0 else 0.0
        self.filt_d_area = (1 - self.alpha) * self.filt_d_area + self.alpha * d_area
        self.prev_area_err = area_err

        lin_vel = self.Kp_lin * area_err + self.Kd_lin * self.filt_d_area
        lin_vel = np.clip(lin_vel, -self.max_lin, self.max_lin)

        # Check for deadband: Are we close enough and stable?
        if abs(area_err) < self.dband and abs(lin_vel) < self.max_lin / 10:
            self.get_logger().warn(
                f'Target {color} ball reached. Localizing...'
            )
            self.move_ttbot(0.0, 0.0)
            
            # --- LOCALIZATION TRIGGER ---
            ox, oy = self.localize_ball(cx)
            self.found_balls[color] = [ox, oy]
            self.get_logger().info(f'Localized {color} ball at: x={ox:.3f}, y={oy:.3f}')
            
            # 1. Move to the next target color
            self.current_target_color_idx += 1
            
            # 2. Advance to the next waypoint in the search sequence
            # We need to manually mark the current goal as 'reached' so the loop picks the next one.
            # This will trigger goal_reached() to be True for the next loop cycle.
            self.goal_pose_given = False
            
            # 3. Clean up the vision window
            cv2.destroyAllWindows() 
            
            self.task3_state = 'SEARCH' # Go back to search for the next ball
            return
        
        # angular PD control (centering)
        cntr_err = fx - cx
        d_cntr = (cntr_err - self.prev_cntr_err) / dt if dt > 0 else 0.0
        self.prev_cntr_err = cntr_err

        ang_vel = self.Kp_ang * cntr_err + self.Kd_ang * d_cntr
        ang_vel = np.clip(ang_vel, -self.max_ang, self.max_ang)
        
        self.move_ttbot(lin_vel, ang_vel)

    # ------------------------------------------------------------------
    # a* planner
    # ------------------------------------------------------------------

    def a_star_path_planner(self, start_pose, end_pose):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info(
            f'A* planner.\n> start: {start_pose.pose.position}\n'
            f'> end:   {end_pose.pose.position}'
        )

        if not self.graph_built:
            self.mp.get_graph_from_map()
            self.graph_built = True
            self.get_logger().info(
                f'Graph rebuilt: {len(self.mp.map_graph.g)} free nodes.'
            )

        s_ix, s_iy = self.convert_coordinates(
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            grid=True
        )
        g_ix, g_iy = self.convert_coordinates(
            end_pose.pose.position.x,
            end_pose.pose.position.y,
            grid=True
        )

        s_key = f'{s_iy},{s_ix}'
        g_key = f'{g_iy},{g_ix}'

        if s_key not in self.mp.map_graph.g or g_key not in self.mp.map_graph.g:
            self.get_logger().warn(
                'Start or goal in obstacle/outside free space. Skipping plan.'
            )
            return path

        self.mp.map_graph.root = s_key
        self.mp.map_graph.end = g_key

        solver = AStar(self.mp.map_graph)

        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        solver.solve(
            self.mp.map_graph.g[s_key],
            self.mp.map_graph.g[g_key]
        )

        self.astarTime = Float32()
        self.astarTime.data = (
            self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        )
        self.calc_time_pub.publish(self.astarTime)

        path_as, dist_as = solver.reconstruct_path(
            self.mp.map_graph.g[s_key],
            self.mp.map_graph.g[g_key]
        )

        for nm in path_as:
            iy, ix = map(int, nm.split(','))
            wx, wy = self.convert_coordinates(ix, iy, grid=False)

            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0

            path.poses.append(ps)

        self.get_logger().info(
            f'A* done (length: {len(path.poses)}, dist: {dist_as:.2f})'
        )

        return path

    # ------------------------------------------------------------------
    # path following
    # ------------------------------------------------------------------

    def get_path_idx(self, path, vehicle_pose):
        n = len(path.poses)
        if n == 0:
            return 0

        self.follow_idx = min(self.follow_idx, n - 1)
        robot_p = vehicle_pose.pose.position

        while self.follow_idx < n - 1:
            wp = path.poses[self.follow_idx].pose.position
            if math.hypot(robot_p.x - wp.x,
                          robot_p.y - wp.y) <= self.wp_reached_thresh:
                self.follow_idx += 1
            else:
                break

        return self.follow_idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """
        pid-based differential-drive path follower
        """

        # constants (tunable)
        MAX_LIN = 0.55
        MAX_ANG = 1.5

        KP_ANG = 3.0
        KI_ANG = 0.05
        KD_ANG = 0.50

        FWD_ERR = 0.3
        ROTATE_ERR = 1.0
        SLOW_RADIUS = 0.6
        STOP_RADIUS = 0.12

        INT_LIM = 0.5

        # timing
        t = self.get_clock().now().nanoseconds * 1e-9
        dt = max(t - self.last_time, 1e-3)
        self.last_time = t

        # extract positions
        rx = vehicle_pose.pose.position.x
        ry = vehicle_pose.pose.position.y
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y

        # heading + distance
        yaw = self.calc_heading(vehicle_pose.pose.orientation)
        desired = math.atan2(gy - ry, gx - rx)
        err = (desired - yaw + math.pi) % (2.0 * math.pi) - math.pi

        dist = math.hypot(gx - rx, gy - ry)

        # narrow passage check
        narrow = False
        if self.map_ready:
            ix, iy = self.convert_coordinates(rx, ry, grid=True)
            prox = self._proximity_inflated(ix, iy, self.clearance_radius_cells)
            narrow = prox > self.narrow_proximity_thresh

        # angular pid
        self.PID_ang_int += err * dt
        self.PID_ang_int = max(-INT_LIM, min(INT_LIM, self.PID_ang_int))

        derr = (err - self.PID_ang_prev) / dt
        self.PID_ang_prev = err

        w = (
            KP_ANG * err
            + KI_ANG * self.PID_ang_int
            + KD_ANG * derr
        )

        w = max(-MAX_ANG, min(MAX_ANG, w))

        # rotate-in-place if badly misaligned
        if abs(err) > ROTATE_ERR:
            return 0.0, w

        # linear speed gating
        heading_scale = max(0.0, 1.0 - abs(err) / ROTATE_ERR)
        dist_scale = min(1.0, dist / max(SLOW_RADIUS, 1e-6))

        v = MAX_LIN * heading_scale * dist_scale

        if narrow:
            v = min(v, self.narrow_lin)

        # move-forward if not too misaligned
        if abs(err) < FWD_ERR:
            return v, 0.0

        # stop near final goal
        if self.path.poses and self.follow_idx >= len(self.path.poses) - 2:
            fp = self.path.poses[-1].pose.position
            if math.hypot(fp.x - rx, fp.y - ry) < STOP_RADIUS:
                self.PID_ang_int = 0.0
                return 0.0, 0.0

        return v, w

    # ------------------------------------------------------------------
    # search helpers
    # ------------------------------------------------------------------

    def goal_reached(self):
        """Checks if the robot is close enough to the current goal pose."""
        if not self.goal_pose_given:
            return True
        
        dx = self.ttbot_pose.pose.position.x - self.goal_pose.pose.position.x
        dy = self.ttbot_pose.pose.position.y - self.goal_pose.pose.position.y
        dist = math.hypot(dx, dy)
        
        # Use a generous threshold for waypoint arrival
        return dist < 0.35 

    def set_next_waypoint_goal(self, x, y):
        """Sets the internal goal_pose to the given waypoint."""
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0 # Default orientation

        self.goal_pose = ps
        self.goal_pose_given = True
        self.plan_computed = False
        self.path = Path() # Clear old path
        
        self.get_logger().info(f'Set new waypoint goal: ({x:.2f}, {y:.2f})')
    
    # ------------------------------------------------------------------
    # main loop + helpers
    # ------------------------------------------------------------------

    def move_ttbot(self, speed, heading):
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = heading
        self.cmd_vel_pub.publish(cmd)

    def loop(self):
        self.publish_inflated_map()

        if len(self.found_balls) >= len(self.target_colors):
            self.task3_state = 'LOCALIZED_ALL'

        if self.task3_state == 'LOCALIZED_ALL':
            self.get_logger().warn('All balls localized. Stopping.')
            self.move_ttbot(0.0, 0.0)
            return

        # ----------------------------------------------------
        # STATE: APPROACH_BALL (Vision-based control)
        # ----------------------------------------------------
        if self.task3_state == 'APPROACH_BALL':
            color = self.target_colors[self.current_target_color_idx]
            self.approach_and_track(color)
            return

        # ----------------------------------------------------
        # STATE: SEARCH (A* planning/following)
        # ----------------------------------------------------
        if self.task3_state == 'SEARCH':
            # if not self.ttbot_pose_given or not self.map_ready or self.cv_img is None:
            #     self.get_logger().info('Waiting for robot pose/map/camera...')
            #     self.move_ttbot(0.0, 0.0)
            #     return

            if not self.ttbot_pose_given:
                self.get_logger().info('Diagnostic: Waiting for Pose/ttbot_pose_given=False')
                self.move_ttbot(0.0, 0.0)
                return
            if not self.map_ready:
                self.get_logger().info('Diagnostic: Waiting for Map/map_ready=False')
                self.move_ttbot(0.0, 0.0)
                return
            if self.cv_img is None:
                self.get_logger().info('Diagnostic: Waiting for Camera/cv_img=None')
                self.move_ttbot(0.0, 0.0)
                return

            current_color = self.target_colors[self.current_target_color_idx]
            
            # 1. Attempt to detect the current target ball
            c, area, center = self.detect_ball(current_color)
            
            if c is not None:
                # Check current navigation state: Only switch to APPROACH if we are moving, 
                # implying the path is clear enough for now.
                # If v is 0, the robot is stuck or avoiding an obstacle (Task 2 logic).
                
                # We need to know the robot's current linear speed. 
                # Since 'v' is local to the path follower, we must calculate it manually
                # or store it in a class member variable.
                # For simplicity, let's rely on the robot's current motion/location.
                
                # Check if the path is currently empty/invalid
                if not self.path.poses or self.follow_idx >= len(self.path.poses):
                    # If we don't have a valid navigation goal, we should transition immediately 
                    # as we are not relying on A* avoidance.
                    pass # Keep the original logic for now

                # Simple check: If the robot is extremely close to the current path waypoint 
                # and still sees the ball, it's a good time to transition.
                # Since the robot is usually moving straight, let's prioritize the transition.
                
                # Reverting to simple transition: relying on __scan_cbk to interrupt APPROACH_BALL 
                # via dynamic map logic if the obstacle is *on the path*.

                # Ball detected! Switch to approach mode.
                self.get_logger().warn(f'Found {current_color} ball! Switching to approach state.')
                # Immediately stop the robot from navigating the A* path
                self.move_ttbot(0.0, 0.0) 
                self.task3_state = 'APPROACH_BALL'
                return
            
            # 2. If no ball is detected, continue standard navigation (Task 2 logic)
            
            # --- Waypoint Management Logic ---
            if not self.initial_search_complete:
                if not self.goal_pose_given or self.goal_reached():
                    # Set the next waypoint as the goal
                    if self.current_wp_idx < len(self.exploration_waypoints):
                        wp_x, wp_y = self.exploration_waypoints[self.current_wp_idx]
                        self.set_next_waypoint_goal(wp_x, wp_y)
                        self.current_wp_idx += 1
                        self.replan = True # Trigger A* planning for the new WP
                    else:
                        self.get_logger().warn('All search waypoints reached. Search complete.')
                        self.initial_search_complete = True
                        self.goal_pose_given = False # Stop following paths

            # Check if planning or replanning is needed
            planning_needed = (self.goal_pose_given and (self.replan or not self.plan_computed))

            # ... (Rest of A* planning and path following logic remains the same) ...
            # ... (The existing code block below goes here) ...

            if planning_needed:
                self.get_logger().info('Planning/Replanning triggered.')
                self.follow_idx = 0

                # This wipes away old dynamic obstacles that are no longer in sight 
                # and were blocking the entire path.
                self.mp.inf_map_img_array = np.copy(self.static_inf_map)

                # ... (A* planning and publication) ...
                # ... (Existing code from your Task 2 loop goes here) ...
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)

                if not path.poses:
                    self.get_logger().warn('Path planner returned an empty path!')
                    self.path = Path()
                    self.plan_computed = False
                    self.replan = False
                    # On plan failure, try next waypoint

                    # --- FIX: New logic to advance waypoint after persistent planning failure ---
                    if self.initial_search_complete:
                        # If all search is done, we are truly stuck.
                        self.get_logger().error('CRITICAL: Cannot reach final destination/waypoint. Stopping.')
                        self.move_ttbot(0.0, 0.0)
                        return
                    
                    # Increment the failure counter and check for persistence
                    self.plan_fail_count += 1
                    if self.plan_fail_count > 3: # If 3 consecutive failures, skip waypoint
                        self.get_logger().warn(f'Waypoint {self.current_wp_idx} unreachable. Skipping to next.')
                        self.current_wp_idx += 1 
                        self.goal_pose_given = False # Force the system to load the next WP goal
                        self.plan_fail_count = 0 
                    
                    # Force replan for next cycle (to try the same goal if count < 3)
                    self.replan = True 
                    return

                # Reset the counter on successful plan
                self.plan_fail_count = 0 
                self.path = path
                self.path_pub.publish(self.path)
                self.plan_computed = True
                self.replan = False

            if self.path.poses:
                # PATH FOLLOWING (Task 2 logic)
                idx = self.get_path_idx(self.path, self.ttbot_pose)

                # ... (rest of path following logic) ...
                if idx >= len(self.path.poses):
                    # Goal Reached! This is handled by the goal_reached() check above
                    # so this block can be simplified/removed.
                    pass # We let the waypoint management handle the transition

                current_goal = self.path.poses[idx]
                v, w = self.path_follower(
                    self.ttbot_pose, current_goal
                )
                self.move_ttbot(v, w)
    
    def run(self):
        self.create_timer(0.1, self.loop)
        rclpy.spin(self)

    # ------------------------------------------------------------------
    # coordinate transforms
    # ------------------------------------------------------------------

    def convert_coordinates(self, x, y, grid=True):
        res = self.mp.map.map_df.resolution[0]
        ox, oy, _ = self.mp.map.map_df.origin[0]
        H, _ = self.mp.map.image_array.shape

        if grid:
            cx = (x - ox) / res
            cy = (y - oy) / res

            ix = int(math.floor(cx))
            iyb = int(math.floor(cy))
            iy = (H - 1) - iyb
            return ix, iy

        ix, iy = x, y
        iyb = (H - 1) - iy

        wx = ox + (ix + 0.5) * res
        wy = oy + (iyb + 0.5) * res
        return wx, wy

    def _proximity_inflated(self, ix, iy, R):
        a = self.mp.inf_map_img_array
        H, W = a.shape

        x0 = max(0, ix - R)
        x1 = min(W - 1, ix + R)
        y0 = max(0, iy - R)
        y1 = min(H - 1, iy + R)

        return float(np.max(a[y0:y1+1, x0:x1+1]))

    def normalize(self, a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def calc_heading(self, q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )

# ======================================================================
# main
# ======================================================================

def main(args=None):
    rclpy.init(args=args)
    nav = Task3()

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()