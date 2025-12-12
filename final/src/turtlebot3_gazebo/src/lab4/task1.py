#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import math
import time
import yaml
import cv2
import heapq
from collections import deque

from geometry_msgs.msg import Quaternion, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray

# ==============================================================
#  A* SUPPORT CLASSES (Tree, TreeNode, AStar)
# ==============================================================

class TreeNode():
    def __init__(self, name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self, node_list, w=None):
        if w is None:
            w = [1] * len(node_list)
        self.children.extend(node_list)
        self.weight.extend(w)

class Tree():
    def __init__(self, name):
        self.name = name
        self.root = ""
        self.end = ""
        self.g = {}  # dict: name -> TreeNode

    def add_node(self, node, start=False, end=False):
        self.g[node.name] = node
        if start:
            self.root = node.name
        elif end:
            self.end = node.name

class AStar():
    def __init__(self, in_tree: Tree):
        import numpy as _np
        self.in_tree = in_tree
        self.dist = {name: _np.inf for name in in_tree.g}
        self.h = {}

        end = tuple(map(int, in_tree.end.split(',')))
        for name in in_tree.g:
            start = tuple(map(int, name.split(',')))
            self.h[name] = _np.hypot(end[0] - start[0], end[1] - start[1])

        self.via = {name: None for name in in_tree.g}
        self.open_set = []   # priority queue: (f_score, counter, node)
        self._counter = 0

    def __get_f_score(self, node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn: TreeNode, en: TreeNode):
        import numpy as _np

        self.dist[sn.name] = 0.0
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

    def reconstruct_path(self, sn: TreeNode, en: TreeNode):
        import numpy as _np

        start_key, end_key = sn.name, en.name
        if self.dist[end_key] == _np.inf:
            return [], _np.inf  # no path found

        path = [end_key]
        while path[-1] != start_key:
            prev = self.via[path[-1]]
            if prev is None:
                return [], _np.inf
            path.append(prev)
        path.reverse()
        return path, self.dist[end_key]

# ==============================================================
#  HELPER FUNCTIONS (map/world conversions + inflation)
# ==============================================================

def world2map(occupancy_grid: OccupancyGrid, xy_tup):
    """
    world (x,y) -> grid (i,j) with i = column, j = row
    Returns (i,j) or None if outside map.
    """
    if occupancy_grid is None:
        return None

    x, y = xy_tup
    origin_x = occupancy_grid.info.origin.position.x
    origin_y = occupancy_grid.info.origin.position.y
    res = occupancy_grid.info.resolution

    i_float = (x - origin_x) / res
    j_float = (y - origin_y) / res

    i_int = int(i_float)
    j_int = int(j_float)

    if (0 <= i_int < occupancy_grid.info.width and
            0 <= j_int < occupancy_grid.info.height):
        return (i_int, j_int)
    else:
        return None

def map2world(occupancy_grid: OccupancyGrid, ij_tup):
    """
    grid (i,j) -> world (x,y) cell center
    """
    if occupancy_grid is None:
        return None

    i, j = ij_tup
    origin_x = occupancy_grid.info.origin.position.x
    origin_y = occupancy_grid.info.origin.position.y
    res = occupancy_grid.info.resolution

    x = origin_x + (i + 0.5) * res
    y = origin_y + (j + 0.5) * res
    return (x, y)

def inflate_obstacles(map_array: np.ndarray, k: int):
    """
    Inflate occupied cells (100) by a k x k square in the grid.

    map_array: np.ndarray with values -1 (unknown), 0 (free), 100 (occupied).
    """
    if map_array is None:
        return None

    rows, cols = map_array.shape
    inflated_map = map_array.copy()

    obstacle_locations = np.argwhere(map_array == 100)
    offset = (k - 1) // 2

    for obs_row, obs_col in obstacle_locations:
        row_start = max(0, obs_row - offset)
        row_end = min(rows, obs_row + offset + 1)
        col_start = max(0, obs_col - offset)
        col_end = min(cols, obs_col + offset + 1)

        region = inflated_map[row_start:row_end, col_start:col_end]
        region[region < 100] += 50

    return inflated_map

# ==============================================================
#  MAIN TASK1 NODE
# ==============================================================

class Task1(Node):
    """
    Frontier-based autonomous mapping for TurtleBot3.
    - Uses live SLAM map on /map
    - Frontier selection (clustered with fallback to single frontier)
    - A* planning on inflated OccupancyGrid
    - Diff-drive path following
    """

    def __init__(self):
        super().__init__('task1_node')

        # --- Params ---
        self.time_limit = 600.0  # seconds
        self.frontier_min_cluster_size = 4
        self.frontier_min_dist_cells = 5  # min distance (in cells) to consider a frontier
        self.inflation_k = 6  # odd number, size of inflation kernel

        # Path following state
        self.path = None  # nav_msgs/Path
        self.follow_idx = 0

        # Robot state (odom)
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # Map state
        self.map_msg = None
        self.grid = None           # raw grid from SLAM
        self.grid_inflated = None  # inflated obstacles
        self.width = None
        self.height = None

        # Time
        self.start_time = time.time()

        # ROS interfaces
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(
            Path, '/task1_path', 10)
        self.frontier_viz_pub = self.create_publisher(
            MarkerArray, "/frontiers", 10)

        # Control loop timer
        self.control_rate = 10.0  # Hz
        self.timer = self.create_timer(1.0 / self.control_rate,
                                       self.control_loop)

        self.get_logger().info("Task1 frontier explorer node initialized.")

    # ----------------------------------------------------------
    #  Callbacks
    # ----------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.width = msg.info.width
        self.height = msg.info.height

        data = np.array(msg.data, dtype=np.int16).reshape((self.height, self.width))
        self.grid = data
        # Inflate obstacles for safer planning
        self.grid_inflated = inflate_obstacles(self.grid, self.inflation_k)

    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        self.robot_x = pose.position.x
        self.robot_y = pose.position.y
        self.robot_yaw = self.calc_heading(pose.orientation)

    # ----------------------------------------------------------
    #  Frontier visualization
    # ----------------------------------------------------------

    def visualize_frontiers(self, frontiers, clusters, centroids):
        """
        frontiers  = list of frontier (i,j) coords
        clusters   = list of clusters (list of frontier cells per cluster)
        centroids  = list of (i,j) centroid positions
        """

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Show individual frontier cells as small cubes (orange)
        for k, (i, j) in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "frontier_points"
            m.id = k
            m.type = Marker.CUBE
            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.05
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 1.0

            x, y = map2world(self.map_msg, (i, j))
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0

            marker_array.markers.append(m)

        # Show cluster centroids as larger spheres (green)
        for k, (i, j) in enumerate(centroids):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "frontier_clusters"
            m.id = 1000 + k
            m.type = Marker.SPHERE
            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0

            x, y = map2world(self.map_msg, (i, j))
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0

            marker_array.markers.append(m)

        self.frontier_viz_pub.publish(marker_array)

    # ----------------------------------------------------------
    #  Utility: heading + angle normalization
    # ----------------------------------------------------------

    def normalize(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def calc_heading(self, q):
        """
        Quaternion -> yaw
        """
        import numpy as _np
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        return _np.arctan2(siny_cosp, cosy_cosp)

    # ----------------------------------------------------------
    #  Main control loop
    # ----------------------------------------------------------

    def control_loop(self):
        now = time.time()

        # Stop on timeout
        if now - self.start_time > self.time_limit:
            self.get_logger().info("Time limit reached. Stopping and saving map.")
            self.stop_robot()
            if self.map_msg is not None:
                self.save_map(self.map_msg, "task1_map")
            rclpy.shutdown()
            return

        # Need map + pose
        if (self.map_msg is None or
                self.grid_inflated is None or
                self.robot_x is None or
                self.robot_y is None):
            return

        # If we currently have a path, follow it
        if self.path is not None and len(self.path.poses) > 0 and self.follow_idx < len(self.path.poses):
            self.follow_current_path()
            return

        # Otherwise, plan a new path to a frontier
        frontier_ij, exploration_complete = self.select_frontier()

        if exploration_complete or frontier_ij is None:
            self.get_logger().info("No more frontiers. Stopping and saving map.")
            self.stop_robot()
            if self.map_msg is not None:
                self.save_map(self.map_msg, "task1_map")
            rclpy.shutdown()
            return

        # Plan using A* from current pose to frontier
        self.plan_path_to_frontier(frontier_ij)

    # ----------------------------------------------------------
    #  Frontier selection (Hybrid: raw grid + inflated planning)
    # ----------------------------------------------------------

    def extract_frontiers(self):
        """
        Returns:
            frontiers = list of (i,j) cells where free → next to unknown
        Uses RAW grid (self.grid) for better detection.
        """
        if self.grid is None:
            return []

        grid = self.grid
        h, w = grid.shape
        frontiers = []

        for j in range(1, h - 1):
            for i in range(1, w - 1):
                cell = grid[j, i]
                # free-ish cell
                if 0 <= cell <= 10:
                    # check 4-neighborhood for unknown
                    if (grid[j-1, i] < 0 or grid[j+1, i] < 0 or
                        grid[j, i-1] < 0 or grid[j, i+1] < 0):
                        frontiers.append((i, j))

        return frontiers

    def cluster_frontiers(self, frontiers, radius=1):
        """
        BFS clustering of frontier cells
        returns -> list of clusters, each cluster = list[(i,j)]
        """
        frontier_set = set(frontiers)
        visited = set()
        clusters = []

        for f in frontiers:
            if f in visited:
                continue

            q = deque([f])
            cluster = [f]
            visited.add(f)

            while q:
                cx, cy = q.popleft()

                for nx in range(cx-radius, cx+radius+1):
                    for ny in range(cy-radius, cy+radius+1):
                        nb = (nx, ny)
                        if nb in visited:
                            continue
                        if nb in frontier_set:
                            visited.add(nb)
                            q.append(nb)
                            cluster.append(nb)

            # enforce minimum cluster size
            if len(cluster) >= self.frontier_min_cluster_size:
                clusters.append(cluster)

        return clusters

    def cluster_centroids(self, clusters):
        """ returns list of (i,j) centroid of each cluster """
        centroids = []
        for cluster in clusters:
            xs = [c[0] for c in cluster]
            ys = [c[1] for c in cluster]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            centroids.append((cx, cy))
        return centroids

    def select_frontier(self):
        if self.grid_inflated is None: 
            return None, False

        grid = self.grid_inflated
        frontiers = []

        h,w = grid.shape

        # find free cells next to unknown
        for j in range(1,h-1):
            for i in range(1,w-1):
                if 0 <= grid[j,i] <= 10:  # free
                    if (grid[j-1,i] < 0 or grid[j+1,i] < 0 or
                        grid[j,i-1] < 0 or grid[j,i+1] < 0):
                        frontiers.append((i,j))

        if not frontiers:
            # no frontiers → mapping done
            if not (grid==-1).any():
                return None, True
            return None, False

        # cluster them
        clusters = self.cluster_frontiers(frontiers)
        centroids = self.cluster_centroids(clusters)

        # visualize frontiers & centroids
        self.visualize_frontiers(frontiers, clusters, centroids)

        # convert robot pose to grid
        start_ij = world2map(self.map_msg,(self.robot_x,self.robot_y))
        if start_ij is None:
            return None, False
        si,sj = start_ij

        best = None
        best_cost = 1e9

        # pick REACHABLE cluster centroid
        for (ci,cj) in centroids:

            # skip if too close to robot ― avoid oscillation
            if math.hypot(ci-si,cj-sj) < self.frontier_min_dist_cells:
                continue

            # skip if inside inflated wall
            if grid[cj,ci] > 10:
                continue

            # evaluate "cost"
            dist = math.hypot(ci-si,cj-sj)

            # info gain — number of unknown cells around frontier
            unknown_count = self.count_unknown(ci,cj,r=4)
            score = dist - unknown_count*2   # reward new areas

            if score < best_cost:
                best_cost = score
                best = (ci,cj)

        if best is None:
            return None, False  # fallback to segment search later

        return best, False

    def count_unknown(self,i,j,r=4):
        grid=self.grid_inflated
        h,w = grid.shape
        cnt=0
        for x in range(i-r,i+r+1):
            for y in range(j-r,j+r+1):
                if 0<=x<w and 0<=y<h and grid[y,x] < 0:
                    cnt+=1
        return cnt

    # ----------------------------------------------------------
    #  A* PLANNING (on inflated grid)
    # ----------------------------------------------------------

    def build_graph_from_grid(self, grid: np.ndarray):
        """
        Build a Tree graph from the inflated occupancy grid:
        - free cells (0<=value<=10) -> nodes
        - 8-connected neighbors -> edges (1 or sqrt(2) weight)
        """
        h, w = grid.shape
        tree = Tree("occ_grid_graph")

        # 1) add all free nodes
        for j in range(h):
            for i in range(w):
                val = grid[j, i]
                if val < 0:
                    continue
                if val > 10:
                    continue
                node = TreeNode(f"{i},{j}")
                tree.add_node(node)

        # 2) connect neighbors
        sqrt2 = math.sqrt(2.0)
        for j in range(h):
            for i in range(w):
                val = grid[j, i]
                if val < 0 or val > 10:
                    continue

                name = f"{i},{j}"
                if name not in tree.g:
                    continue
                node = tree.g[name]

                # neighbors (8-connected)
                for di, dj, wgt in [
                    (1, 0, 1.0),
                    (-1, 0, 1.0),
                    (0, 1, 1.0),
                    (0, -1, 1.0),
                    (1, 1, sqrt2),
                    (1, -1, sqrt2),
                    (-1, 1, sqrt2),
                    (-1, -1, sqrt2)
                ]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < w and 0 <= nj < h:
                        nval = grid[nj, ni]
                        if 0 <= nval <= 10:
                            cname = f"{ni},{nj}"
                            if cname in tree.g:
                                node.add_children([tree.g[cname]], [wgt])

        return tree

    def plan_path_to_frontier(self, frontier_ij):
        """
        A* path to a frontier cell (i,j).
        Uses nearest-free fallback for both start & goal if inflated cells block them.
        """

        if self.map_msg is None or self.grid_inflated is None:
            return

        # ---------------------------------------------------------
        # Robot pose in grid coords
        start_ij = world2map(self.map_msg,(self.robot_x,self.robot_y))
        if start_ij is None:
            self.get_logger().warn("Robot is outside grid bounds.")
            return
        si,sj = start_ij   # (i,j)

        # Frontier goal from select_frontier()
        gi,gj = frontier_ij

        # Safety bounds
        if not (0 <= gi < self.width and 0 <= gj < self.height):
            self.get_logger().warn("Frontier goal outside grid.")
            return

        grid = self.grid_inflated

        # ---------------------------------------------------------
        # If robot is inside inflation, find nearest free
        if grid[sj,si] > 10:
            si,sj = self.nearest_free(si,sj, grid)
            if si is None:
                self.get_logger().warn("Start position fully blocked.")
                return

        # If frontier centroid hit inflation, correct it
        if grid[gj,gi] > 10:
            gi,gj = self.nearest_free(gi,gj, grid)
            if gi is None:
                self.get_logger().warn("Frontier unreachable — skipping.")
                return

        # ---------------------------------------------------------
        # Build graph
        tree = self.build_graph_from_grid(grid)

        start_name = f"{si},{sj}"
        goal_name  = f"{gi},{gj}"

        if start_name not in tree.g or goal_name not in tree.g:
            self.get_logger().warn("A* start/goal not free in graph.")
            return

        # MUST set these BEFORE AStar()
        tree.root = start_name
        tree.end  = goal_name

        start_node = tree.g[start_name]
        goal_node  = tree.g[goal_name]

        # ---------------------------------------------------------
        # Run A*
        self.get_logger().info(f"Running A* from {start_name} → {goal_name}")
        t0=time.time()
        astar = AStar(tree)
        astar.solve(start_node,goal_node)
        path_names,dist = astar.reconstruct_path(start_node,goal_node)

        if not path_names:
            self.get_logger().warn("A* returned empty path.")
            return

        # ---------------------------------------------------------
        # Convert to nav_msgs/Path
        path=Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id="map"

        for name in path_names:
            i,j = map(int,name.split(","))
            wx,wy = map2world(self.map_msg,(i,j))
            ps=PoseStamped()
            ps.header=path.header
            ps.pose.position.x=wx
            ps.pose.position.y=wy
            path.poses.append(ps)

        self.path=path
        self.follow_idx=0
        self.path_pub.publish(path)
        self.get_logger().info(f"Path OK ({len(path.poses)} pts). Following...")
    
    def nearest_free(self,i,j,grid):
        """Return nearest free cell (0-10) around (i,j)"""
        for r in range(1,15):
            for dx in range(-r,r+1):
                for dy in range(-r,r+1):
                    nx=i+dx; ny=j+dy
                    if 0<=nx<self.width and 0<=ny<self.height:
                        if 0<=grid[ny,nx]<=10:
                            return nx,ny
        return None,None

    # ----------------------------------------------------------
    #  PATH FOLLOWING
    # ----------------------------------------------------------

    def get_path_idx(self, path: Path, vehicle_pose: PoseStamped):
        """
        Path index progression logic: move to next waypoint when close.
        """
        ADVANCE_DIST_THRESHOLD = 0.2

        n = len(path.poses)
        if n == 0:
            return 0

        self.follow_idx = min(self.follow_idx, n - 1)

        vx = vehicle_pose.pose.position.x
        vy = vehicle_pose.pose.position.y

        while self.follow_idx < n - 1:
            wp = path.poses[self.follow_idx].pose.position
            dist = np.hypot(vx - wp.x, vy - wp.y)
            if dist <= ADVANCE_DIST_THRESHOLD:
                self.follow_idx += 1
            else:
                break

        return self.follow_idx

    def path_follower(self, vehicle_pose: PoseStamped, current_goal_pose: PoseStamped):
        """
        Heading + distance-based path follower.
        Returns (speed, heading_rate).
        """
        MAX_LIN_SPEED = 1.6
        MAX_ANG_SPEED = 1.0
        THRESHOLD = 0.15
        ANG_GAIN = 1.2
        SCALE_DIST = 1.8
        STOP_DIST = 0.2

        vx = vehicle_pose.pose.position.x
        vy = vehicle_pose.pose.position.y
        tx = current_goal_pose.pose.position.x
        ty = current_goal_pose.pose.position.y

        vhead = self.calc_heading(vehicle_pose.pose.orientation)
        thead = np.arctan2(ty - vy, tx - vx)
        err_head = self.normalize(thead - vhead)

        err_dist = np.hypot(tx - vx, ty - vy)

        # rotate-in-place if facing away
        if abs(err_head) > THRESHOLD:
            speed = 0.0
            heading = float(np.clip(ANG_GAIN * err_head, -MAX_ANG_SPEED, MAX_ANG_SPEED))
            return speed, heading

        scale_heading = max(0.0, 0.5 + 0.5 * np.cos(err_head))
        scale_dist = min(1.0, err_dist / max(SCALE_DIST, 1e-6))
        speed = MAX_LIN_SPEED * scale_heading * scale_dist
        heading = 0.0

        if self.path and self.follow_idx >= len(self.path.poses) - 2:
            final_p = self.path.poses[-1].pose.position
            final_dist = np.hypot(final_p.x - vx, final_p.y - vy)
            if final_dist < STOP_DIST:
                self.follow_idx = len(self.path.poses)
                return 0.0, 0.0

        speed = float(max(min(speed, MAX_LIN_SPEED), -MAX_LIN_SPEED))
        heading = float(max(min(heading, MAX_ANG_SPEED), -MAX_ANG_SPEED))

        return speed, heading

    def follow_current_path(self):
        """
        Called from control_loop when self.path is active.
        """
        if (self.robot_x is None or
                self.robot_y is None or
                self.robot_yaw is None or
                self.path is None or
                len(self.path.poses) == 0):
            return

        # build vehicle PoseStamped in map frame
        vehicle_pose = PoseStamped()
        vehicle_pose.header.frame_id = "map"
        vehicle_pose.header.stamp = self.get_clock().now().to_msg()
        vehicle_pose.pose.position.x = float(self.robot_x)
        vehicle_pose.pose.position.y = float(self.robot_y)
        vehicle_pose.pose.position.z = 0.0

        # yaw -> quaternion
        cy = math.cos(self.robot_yaw * 0.5)
        sy = math.sin(self.robot_yaw * 0.5)
        vehicle_pose.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=sy, w=cy
        )

        idx = self.get_path_idx(self.path, vehicle_pose)
        if idx >= len(self.path.poses):
            self.stop_robot()
            self.path = None
            return

        goal_pose = self.path.poses[idx]
        speed, heading = self.path_follower(vehicle_pose, goal_pose)
        self.move_ttbot(speed, heading)

    # ----------------------------------------------------------
    #  LOW-LEVEL MOTION + STOP
    # ----------------------------------------------------------

    def move_ttbot(self, speed, heading):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(heading)
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.move_ttbot(0.0, 0.0)

    # ----------------------------------------------------------
    #  MAP SAVE (.pgm + .yaml)
    # ----------------------------------------------------------

    def save_map(self, map_msg: OccupancyGrid, name="task1_map"):
        """
        Convert OccupancyGrid to PGM + YAML.
        """
        self.get_logger().info(f"Saving map as {name}.pgm / {name}.yaml")

        width = map_msg.info.width
        height = map_msg.info.height
        res = map_msg.info.resolution
        origin = map_msg.info.origin

        data = np.array(map_msg.data, dtype=np.int16).reshape((height, width))

        img = np.zeros((height, width), dtype=np.uint8)

        # typical convention: 0 = occupied (black), 255 = free (white), 205 = unknown
        img[data == 100] = 0       # occupied
        img[data == 0] = 255       # free
        img[data < 0] = 205        # unknown

        pgm_name = f"{name}.pgm"
        yaml_name = f"{name}.yaml"

        # Save PGM (OpenCV writes PGM if extension is .pgm)
        cv2.imwrite(pgm_name, img)

        # Save YAML
        map_yaml = {
            'image': pgm_name,
            'resolution': float(res),
            'origin': [float(origin.position.x), float(origin.position.y), 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196
        }

        with open(yaml_name, 'w') as f:
            yaml.dump(map_yaml, f, default_flow_style=False)

        self.get_logger().info("Map saved.")

# ==============================================================
#  MAIN
# ==============================================================

def main(args=None):
    rclpy.init(args=args)
    node = Task1()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()