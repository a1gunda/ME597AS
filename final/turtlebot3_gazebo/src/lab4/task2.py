#!/usr/bin/env python3
import os
import cv2
import yaml
import rclpy
import heapq
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from copy import copy
from rclpy.node import Node
from PIL import Image, ImageOps
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

## from lab 3: auto_navigator.py
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
        im = Image.open(os.path.join(map_path, map_name))
        im = ImageOps.grayscale(im)
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

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self, nodes, w=None):
        if w is None:
            w = [1] * len(nodes)
        self.children.extend(nodes)
        self.weight.extend(w)

class Tree:
    def __init__(self, name):
        self.name = name
        self.root = None
        self.end = None
        self.g = {}
        self.h_cache = {}

    def add_node(self, node, start=False, end=False):
        self.g[node.name] = node
        if start:
            self.root = node.name
        if end:
            self.end = node.name

    def clear_heuristic(self):
        self.h_cache = {}

class AStar:
    def __init__(self, in_tree):
        self.in_tree = in_tree

        self.dist = {k: np.inf for k in in_tree.g}
        self.via = {k: None for k in in_tree.g}

        if not in_tree.h_cache:
            end = tuple(map(int, in_tree.end.split(',')))
            for name in in_tree.g:
                start = tuple(map(int, name.split(',')))
                in_tree.h_cache[name] = np.hypot(
                    end[0] - start[0],
                    end[1] - start[1]
                )

        self.h = in_tree.h_cache
        self.open_set = []
        self._counter = 0

    def _f(self, node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0.0
        heapq.heappush(
            self.open_set,
            (self._f(sn), self._counter, sn)
        )
        self._counter += 1

        visited = set()
        best_goal_dist = np.inf

        while self.open_set:
            _, __, u = heapq.heappop(self.open_set)

            if u.name in visited:
                continue
            visited.add(u.name)

            if u.name == en.name:
                best_goal_dist = self.dist[u.name]
                break

            if self.dist[u.name] + self.h[u.name] > best_goal_dist:
                break

            du = self.dist[u.name]
            children = u.children
            weights = u.weight

            for c, w in zip(children, weights):
                nd = du + w
                if nd < self.dist[c.name]:
                    self.dist[c.name] = nd
                    self.via[c.name] = u.name
                    heapq.heappush(
                        self.open_set,
                        (nd + self.h[c.name], self._counter, c)
                    )
                    self._counter += 1

    def reconstruct_path(self, sn, en):
        start_key = sn.name
        end_key = en.name

        if self.dist[end_key] == np.inf:
            return [], np.inf

        path = [end_key]
        while path[-1] != start_key:
            prev = self.via[path[-1]]
            if prev is None:
                return [], np.inf
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

## task 2 node
class Task2(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self, node_name='task2_node'):
        super().__init__(node_name)

        # ros interface
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(LaserScan, '/scan', self.__scan_cbk, 10)

        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time', 10)
        self.inf_map_pub = self.create_publisher(OccupancyGrid, '/inflated_map', 10)

        # core state
        self.path, self.goal_pose, self.ttbot_pose = Path(), PoseStamped(), PoseStamped()
        self.follow_idx, self.start_time = 0, 0.0

        # flags
        self.goal_pose_given = self.ttbot_pose_given = False
        self.plan_computed = self.replan = False

        # replanning / hysteresis
        self.last_replan_time = 0.0
        self.replan_cooldown = 1.5
        self.min_commit_idx = 8
        self.dyn_obstacle_lifetime = 5.0
        self.dyn_obstacles = {}

        # path tracking / safety
        self.wp_reached_thresh = 0.25
        self.clear_radius_cells = 4
        self.narrow_prox_thresh = 0.15
        self.narrow_lin = 0.15

        # map / planner setup
        self.map_name = 'map'
        self.dyn_kr, self.kr = 6, 10

        self.mp_dyn = MapProcessor(self.map_name)
        self.mp_dyn.inflate_map(self.mp_dyn.rect_kernel(self.dyn_kr, 1), True)

        self.mp = MapProcessor(self.map_name)
        self.mp.inflate_map(self.mp.rect_kernel(self.kr, 1), True)
        self.mp.get_graph_from_map()

        # cached map metadata
        self.res = self.mp.map.map_df.resolution[0]
        self.origin = self.mp.map.map_df.origin[0]
        self.shape = self.mp.map.image_array.shape[0]

        self.graph_built = self.map_ready = True
        self.inf_map_stat = np.copy(self.mp.inf_map_img_array)

        h, w = self.mp.map.image_array.shape
        self.get_logger().info(f'Local map loaded: {w}x{h}, nodes={len(self.mp.map_graph.g)}')

    ## callbacks
    def __goal_pose_cbk(self, msg):
        # new goal received
        self.goal_pose = msg
        self.goal_pose.header.frame_id = 'map'
        self.goal_pose_given = True

        # reset dynamic + planning maps
        self.mp_dyn = MapProcessor(self.map_name)
        self.mp_dyn.inflate_map(self.mp_dyn.rect_kernel(self.dyn_kr, 1), True)

        self.mp = MapProcessor(self.map_name)
        self.mp.inflate_map(self.mp.rect_kernel(self.kr, 1), True)
        self.mp.map_graph.clear_heuristic()

        self.graph_built = False
        self.replan = True
        self.plan_computed = False

        self.get_logger().info(f'goal_pose: {self.goal_pose.pose.position.x:.4f}, {self.goal_pose.pose.position.y:.4f}')

    def __ttbot_pose_cbk(self, msg):
        # amcl pose update
        self.ttbot_pose.header = msg.header
        self.ttbot_pose.pose = msg.pose.pose
        self.ttbot_pose.header.frame_id = 'map'
        self.ttbot_pose_given = True

        self.get_logger().info(f'ttbot_pose: {self.ttbot_pose.pose.position.x:.4f}, {self.ttbot_pose.pose.position.y:.4f}')

    def __scan_cbk(self, msg):
        # lidar-based dynamic obstacles + replanning trigger
        if not self.ttbot_pose_given or not self.map_ready:
            return

        # robot pose in map frame
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = self.calc_heading(self.ttbot_pose.pose.orientation)

        # filter useful lidar returns
        ranges = np.array(msg.ranges)
        valid_idx = np.where((ranges < 1.5) & (ranges > 0.2))[0]

        # map dimensions and inflation radii
        H, W = self.mp.inf_map_img_array.shape
        kr_dyn, kr_inf = self.dyn_kr // 2, self.kr // 5
        map_changed = False

        # project lidar hits into grid and mark as dynamic obstacles
        for i in valid_idx:
            r = ranges[i]
            theta = msg.angle_min + i * msg.angle_increment

            ox = rx + r * np.cos(yaw + theta)
            oy = ry + r * np.sin(yaw + theta)

            ix, iy = self.convert_coordinates(ox, oy, grid=True)
            if not (0 <= ix < W and 0 <= iy < H):
                continue

            # skip if already occupied in static inflated map
            if self.inf_map_stat[iy, ix] > 0:
                continue

            map_changed = True
            self.dyn_obstacles[(ix, iy)] = self.get_clock().now().nanoseconds * 1e-9

        if map_changed:
            # expire old dynamic obstacles to avoid map pollution
            t = self.get_clock().now().nanoseconds * 1e-9
            self.dyn_obstacles = {k: v for k, v in self.dyn_obstacles.items() if t - v < self.dyn_obstacle_lifetime}

            # rebuild inflated maps with current dynamic obstacles
            self.mp_dyn.inf_map_img_array[:] = 0
            self.mp.inf_map_img_array[:] = self.inf_map_stat

            for ix, iy in self.dyn_obstacles:
                self.inflate_cell(self.mp_dyn.inf_map_img_array, ix, iy, kr_dyn)
                self.inflate_cell(self.mp.inf_map_img_array, ix, iy, kr_inf)

        # check near-term waypoints for newly blocked cells
        if map_changed and self.path.poses:
            k0, k1 = self.follow_idx, min(len(self.path.poses), self.follow_idx + 20)

            for k in range(k0, k1):
                wp = self.path.poses[k].pose.position
                wx, wy = self.convert_coordinates(wp.x, wp.y, grid=True)

                # collision detected along committed path segment
                if 0 <= wx < W and 0 <= wy < H and self.mp.inf_map_img_array[wy, wx] > 0:
                    t = self.get_clock().now().nanoseconds * 1e-9

                    # hysteresis: avoid thrashing on early or recent replans
                    if self.follow_idx < self.min_commit_idx or t - self.last_replan_time < self.replan_cooldown:
                        return

                    self.get_logger().warn('Path Blocked by Dynamic Obstacle! Replanning...')
                    self.move_ttbot(0.0, 0.0)

                    # force fresh planning from current pose
                    self.path = Path()
                    self.replan = True
                    self.graph_built = False
                    self.plan_computed = False
                    break

    ## visualization
    def publish_inflated_map(self):
        # publish inflated occupancy grid for visualization
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

        # flip for ros coords and binarize to occupancy values
        arr = np.flipud(self.mp.inf_map_img_array)
        msg.data = np.where(arr.flatten() > 0, 100, 0).astype(np.int8).tolist()

        self.inf_map_pub.publish(msg)

    ## planning
    def a_star_path_planner(self, start_pose, end_pose):
        # grid-based global planning using inflated occupancy map
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info(
            f'A* planner.\n> start: {start_pose.pose.position}\n'
            f'> end:   {end_pose.pose.position}'
        )

        # rebuild graph if map has changed
        if not self.graph_built:
            self.mp.get_graph_from_map()
            self.graph_built = True
            self.get_logger().info(f'Graph rebuilt: {len(self.mp.map_graph.g)} free nodes.')

        # convert start and goal to grid coordinates
        s_ix, s_iy = self.convert_coordinates(start_pose.pose.position.x, start_pose.pose.position.y, grid=True)
        g_ix, g_iy = self.convert_coordinates(end_pose.pose.position.x, end_pose.pose.position.y, grid=True)

        # snap start and goal to nearest free cells
        s_free = self.find_nearest_free(s_ix, s_iy)
        g_free = self.find_nearest_free(g_ix, g_iy)
        if s_free is None or g_free is None:
            self.get_logger().warn('No free start or goal cell found.')
            return path

        s_ix, s_iy = s_free
        g_ix, g_iy = g_free
        s_key, g_key = f'{s_iy},{s_ix}', f'{g_iy},{g_ix}'

        # validate nodes against graph
        if s_key not in self.mp.map_graph.g or g_key not in self.mp.map_graph.g:
            self.get_logger().warn('Start or goal invalid (occupied or outside map).')
            return path

        self.mp.map_graph.root = s_key
        self.mp.map_graph.end = g_key

        # run A* on grid graph
        solver = AStar(self.mp.map_graph)
        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        solver.solve(self.mp.map_graph.g[s_key], self.mp.map_graph.g[g_key])

        self.calc_time_pub.publish(Float32(
            data=self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        ))

        # reconstruct path and convert back to world coordinates
        path_as, dist_as = solver.reconstruct_path(self.mp.map_graph.g[s_key], self.mp.map_graph.g[g_key])
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

        self.get_logger().info(f'A* done (length={len(path.poses)}, dist={dist_as:.2f})')
        return path

    def find_nearest_free(self, ix, iy, max_radius=8):
        # search outward until a free grid cell is found
        a = self.mp.inf_map_img_array
        H, W = a.shape

        if 0 <= ix < W and 0 <= iy < H and a[iy, ix] == 0:
            return ix, iy

        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < W and 0 <= ny < H and a[ny, nx] == 0:
                        return nx, ny

        return None

    ## path following
    def get_path_idx(self, path, vehicle_pose):
        # advance along path as waypoints are reached
        n = len(path.poses)
        if n == 0:
            return 0

        self.follow_idx = min(self.follow_idx, n - 1)
        rp = vehicle_pose.pose.position

        while self.follow_idx < n - 1:
            wp = path.poses[self.follow_idx].pose.position
            if np.hypot(rp.x - wp.x, rp.y - wp.y) <= self.wp_reached_thresh:
                self.follow_idx += 1
            else:
                break

        return self.follow_idx

    def path_follower(self, vehicle_pose, current_goal_pose=None):
        # curvature-based path tracking with adaptive lookahead
        MAX_LIN, MAX_ANG = 0.65, 2.1
        Ld_min, Ld_max = 0.35, 0.85
        STOP_RADIUS = 0.18

        rx = vehicle_pose.pose.position.x
        ry = vehicle_pose.pose.position.y
        yaw = self.calc_heading(vehicle_pose.pose.orientation)

        # stop when close enough to final waypoint
        fp = self.path.poses[-1].pose.position
        goal_dist = np.hypot(fp.x - rx, fp.y - ry)
        if goal_dist < STOP_RADIUS:
            return 0.0, 0.0

        # scale lookahead based on remaining distance
        speed_ratio = min(1.0, goal_dist / 1.0)
        Ld = Ld_min + (Ld_max - Ld_min) * speed_ratio
        Ld = max(Ld, 0.5 * Ld_max)

        # reduce lookahead in high-curvature segments
        if self.follow_idx < len(self.path.poses) - 2:
            p0 = self.path.poses[self.follow_idx].pose.position
            p1 = self.path.poses[self.follow_idx + 1].pose.position
            p2 = self.path.poses[self.follow_idx + 2].pose.position

            v1x, v1y = p1.x - p0.x, p1.y - p0.y
            v2x, v2y = p2.x - p1.x, p2.y - p1.y
            ang = abs(np.arctan2(v2x * v1y - v2y * v1x, v2x * v1x + v2y * v1y))

            if ang > 0.35:
                Ld *= 0.6

        # compute lookahead target
        lookahead = self.get_lookahead_point(self.path, vehicle_pose, Ld)
        dx, dy = lookahead.x - rx, lookahead.y - ry

        # transform target into robot frame
        x_r = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        y_r = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        # rotate in place if target is mostly behind
        desired = np.arctan2(y_r, x_r)
        if abs(desired) > np.pi / 2:
            return 0.0, np.clip(2.1 * desired, -MAX_ANG, MAX_ANG)

        # small forward bias to escape local oscillations
        if x_r <= 0.05:
            return 0.1, np.clip(2.0 * np.sign(y_r), -MAX_ANG, MAX_ANG)

        # pure-pursuit curvature
        curvature = 2.0 * y_r / (Ld ** 2)
        R_min = 0.25
        curvature = np.clip(curvature, -1.0 / R_min, 1.0 / R_min)

        # reduce speed as curvature increases
        heading_scale = np.clip(1.0 - 1.5 * abs(curvature) * Ld, 0.15, 1.0)
        v = MAX_LIN * heading_scale

        # slow down in narrow or inflated regions
        if self.map_ready:
            ix, iy = self.convert_coordinates(rx, ry, grid=True)
            if self.proximity_inflated(ix, iy, self.clear_radius_cells) > self.narrow_prox_thresh:
                v = min(v, self.narrow_lin)

        w = curvature * v
        return np.clip(v, 0.05, MAX_LIN), np.clip(w, -MAX_ANG, MAX_ANG)

    def get_lookahead_point(self, path, pose, Ld):
        # pick first waypoint ahead of robot at distance >= Ld
        rx = pose.pose.position.x
        ry = pose.pose.position.y

        k0 = min(self.follow_idx, len(path.poses) - 2)
        p0 = path.poses[k0].pose.position
        p1 = path.poses[k0 + 1].pose.position

        # local path tangent for forward filtering
        tx, ty = p1.x - p0.x, p1.y - p0.y
        tnorm = np.hypot(tx, ty)
        if tnorm > 1e-6:
            tx, ty = tx / tnorm, ty / tnorm

        for k in range(self.follow_idx, len(path.poses)):
            wp = path.poses[k].pose.position
            dx, dy = wp.x - rx, wp.y - ry

            if np.hypot(dx, dy) < Ld:
                continue
            if dx * tx + dy * ty < 0.0:
                continue

            return wp

        return path.poses[-1].pose.position

    def move_ttbot(self, speed, heading):
        # publish velocity command
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = heading
        self.cmd_vel_pub.publish(cmd)

    ## main loop
    def loop(self):
        # visualize current inflated map
        self.publish_inflated_map()

        # trigger planning or replanning
        if (self.map_ready and self.ttbot_pose_given and self.goal_pose_given and
                (self.replan or not self.plan_computed)):

            self.get_logger().info('Planning/Replanning triggered.')
            self.follow_idx = 0
            self.last_replan_time = self.get_clock().now().nanoseconds * 1e-9

            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            if not path.poses:
                self.get_logger().warn('Path planner returned an empty path!')
                self.path = Path()
                self.plan_computed = False
                self.replan = False
                return

            self.path = path
            self.path_pub.publish(self.path)
            self.plan_computed = True
            self.replan = False

            self.get_logger().info('Path has been generated')
            self.get_logger().info(f'Number of poses on path: {len(self.path.poses)}')

        # follow current path if available
        if self.path.poses:
            idx = self.get_path_idx(self.path, self.ttbot_pose)
            self.get_logger().info(f'Current path index: {idx}')

            # stop once final waypoint is reached
            if idx >= len(self.path.poses) - 1:
                self.get_logger().info('Goal Pose Reached!')
                self.move_ttbot(0.0, 0.0)
                self.goal_pose_given = False
                self.plan_computed = False
                return

            # track path using local controller
            current_goal = self.path.poses[idx]
            v, w = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(v, w)

    def run(self):
        # periodic control loop
        self.create_timer(0.1, self.loop)
        rclpy.spin(self)

    ## helpers
    def inflate_cell(self, a, ix, iy, R, value=100):
        # inflate a square region around a grid cell
        H, W = a.shape

        # clamp inflation bounds to map limits
        x0, x1 = max(0, ix - R), min(W, ix + R + 1)
        y0, y1 = max(0, iy - R), min(H, iy + R + 1)

        # mark inflated region as occupied
        a[y0:y1, x0:x1] = value

    def proximity_inflated(self, ix, iy, R):
        # measure maximum inflated occupancy around a cell
        a = self.mp.inf_map_img_array
        H, W = a.shape

        # extract local neighborhood
        x0, x1 = max(0, ix - R), min(W - 1, ix + R)
        y0, y1 = max(0, iy - R), min(H - 1, iy + R)

        # return proximity indicator (0 free, >0 occupied)
        return float(np.max(a[y0:y1 + 1, x0:x1 + 1]))

    def convert_coordinates(self, x, y, grid=True):
        # convert between world and grid coordinates
        res = self.mp.map.map_df.resolution[0]
        ox, oy, _ = self.mp.map.map_df.origin[0]
        H, _ = self.mp.map.image_array.shape

        if grid:
            # world -> grid (account for flipped y-axis)
            cx = (x - ox) / res
            cy = (y - oy) / res

            ix = int(np.floor(cx))
            iy = (H - 1) - int(np.floor(cy))
            return ix, iy

        # grid -> world (cell center)
        ix, iy = x, y
        iyb = (H - 1) - iy

        wx = ox + (ix + 0.5) * res
        wy = oy + (iyb + 5e-1) * res
        return wx, wy

    def normalize(self, a):
        # wrap angle to [-pi, pi]
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def calc_heading(self, q):
        # extract yaw from quaternion
        return np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )
    
## main
def main(args=None):
    rclpy.init(args=args)
    task2 = Task2()

    try:
        task2.run()
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
