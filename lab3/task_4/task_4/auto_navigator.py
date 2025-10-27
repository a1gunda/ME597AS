#!/usr/bin/env python3
import os
import cv2
import yaml
import time
import rclpy
import heapq
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from copy import copy
from rclpy.node import Node
from nav_msgs.msg import Path
from PIL import Image, ImageOps
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

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
        map_path = os.path.expanduser('~/Documents/ros2_ws/src/task_4/maps')
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

class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Path planner/follower related variables
        self.path = None
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        self.follow_idx = 0
        self.goal_pose_given = False
        self.ttbot_pose_given = False

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY

        # Node rate
        self.rate = self.create_rate(10)

        # Map generation
        mp = MapProcessor('sync_classroom_map')
        kr = mp.rect_kernel(12,1)
        mp.inflate_map(kr,True)
        mp.get_graph_from_map()
        self.res = mp.map.map_df['resolution'][0]
        self.origin = mp.map.map_df['origin'][0]
        self.shape = mp.map.map_cv2.shape[0]
        self.mp = mp

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose_given = True
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose_given = True
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        # define path
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        
        # start position
        mp = self.mp
        x,y = self.convert_coordinates(self, start_pose.pose.position.x, start_pose.pose.position.y)
        mp.map_graph.root = f"{y},{x}"
        start_node = mp.map_graph.g[mp.map_graph.root]
        
        # end position
        x,y = self.convert_coordinates(self, end_pose.pose.position.x, end_pose.pose.position.y)
        mp.map_graph.end = f"{y},{x}"
        end_node = mp.map_graph.g[mp.map_graph.end]
        
        # run A*
        as_maze = AStar(mp.map_graph)
        self.get_logger().info('Running A*...')
        start = time.time()
        as_maze.solve(start_node,end_node)     
        end = time.time()
        self.get_logger().info(f"A* Solved in {(end-start):.8f}s")
        path_as,dist_as = as_maze.reconstruct_path(start_node,end_node)

        # update path
        for pose in path_as:
            temp = PoseStamped()
            temp.header = path.header
            y,x = map(float, pose.split(','))
            cx,cy = self.convert_coordinates(self,x,y,grid=False)
            temp.pose.position.x = cx
            temp.pose.position.y = cy
            path.poses.append(temp)
        
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        return path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        # constant
        ADVANCE_DIST_THRESHOLD = 0.2    # [m] distance within which to consider a waypoint "reached"

        # empty path
        n = len(path.poses)
        if n == 0:
            return 0

        # check current follow index within valid bounds
        self.follow_idx = min(self.follow_idx, n - 1)

        # advance to the next waypoint if the robot is close to the current one
        vehicle = vehicle_pose.pose.position
        while self.follow_idx < n - 1:
            waypoint = path.poses[self.follow_idx].pose.position
            dist = np.hypot(vehicle.x - waypoint.x, vehicle.y - waypoint.y)
            if dist <= ADVANCE_DIST_THRESHOLD:
                self.follow_idx += 1  # move to next waypoint
            else:
                break  # still too far from current waypoint

        return self.follow_idx
    
    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        # constants
        MAX_LIN_SPEED = 0.6             # [m/s] maximum forward speed
        MAX_ANG_SPEED = 0.4             # [rad/s] maximum angular speed
        THRESHOLD = 0.2                 # [rad] threshold to rotate in place
        ANG_GAIN = 1.2                  # proportional gain for angular correction
        SCALE_DIST = 1.8                # distance for full linear speed scaling
        STOP_DIST = 0.2                 # [m] stop threshold near final goal

        # extract positions
        vx, vy = vehicle_pose.pose.position.x, vehicle_pose.pose.position.y
        tx, ty = current_goal_pose.pose.position.x, current_goal_pose.pose.position.y

        # heading error
        vhead = self.calc_heading(vehicle_pose.pose.orientation)
        thead = np.arctan2(ty - vy, tx - vx)
        err_head = self.normalize(thead - vhead)

        # distance error
        err_dist = np.hypot(tx - vx, ty - vy)

        # rotate-in-place if facing far from goal direction
        if abs(err_head) > THRESHOLD:
            speed = 0.0
            heading = np.clip(ANG_GAIN * err_head, -MAX_ANG_SPEED, MAX_ANG_SPEED)
            
            return speed, heading

        # linear velocity scaling
        scale_heading = max(0.0, 0.5 + 0.5 * np.cos(err_head))
        scale_dist = min(1.0, err_dist / max(SCALE_DIST, 1e-6))
        speed = MAX_LIN_SPEED * scale_heading * scale_dist
        heading = 0.0

        # stop when near final global goal
        if self.path and self.follow_idx >= len(self.path.poses) - 2:
            final_p = self.path.poses[-1].pose.position
            final_dist = np.hypot(final_p.x - vx, final_p.y - vy)
            if final_dist < STOP_DIST:
                self.follow_idx = len(self.path.poses)
                return 0.0, 0.0

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading
        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        while rclpy.ok():
            rclpy.spin_once(self)

            # 1: wait for valid goal pose
            if not self.goal_pose_given:
                self.get_logger().info("Waiting for goal pose to be received...")
                continue

            # 2: wait for valid current pose
            if not self.ttbot_pose_given:
                continue

            # ensure goal pose is valid (not starting pose)
            if (self.goal_pose.pose.position.x == self.ttbot_pose.pose.position.x and 
                self.goal_pose.pose.position.y == self.ttbot_pose.pose.position.y):
                self.get_logger().warn('Invalid goal pose.')
                self.goal_pose_given = False
                continue

            # reset flag to prevent repeated processing of the same pose
            self.ttbot_pose_given = False

            # 3: generate path if not yet created
            if self.path is None:
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                if path is None or len(path.poses) == 0:
                    self.get_logger().warn("Path planner returned an empty path!")
                    continue

                self.path_pub.publish(path)
                self.get_logger().info('Path has been generated')
                self.get_logger().info(f'Number of poses on path: {len(path.poses)}')
                self.path = path
            else:
                path = self.path  # use existing path

            # 4: determine next waypoint to follow
            idx = self.get_path_idx(path, self.ttbot_pose)
            self.get_logger().info(f'Current path index: {idx}')

            # 5: compute control commands and move the robot
            current_goal = path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)

            # 6: check if goal reached
            if idx >= len(path.poses):
                self.get_logger().info('Goal Pose Reached!')
                self.move_ttbot(0.0, 0.0)
                continue

            self.rate.sleep()

    @staticmethod
    def convert_coordinates(self, x, y, grid=True):
        if grid:
            cx = int((x - self.origin[0]) / self.res)
            cy = int((self.origin[1] + self.shape*self.res - y) / self.res)
        else:
            cx = (x * self.res) + self.origin[0]
            cy = (self.shape * self.res + self.origin[1]) - (y * self.res)
        return cx, cy

    @staticmethod
    def normalize(a):
        return (a + np.pi) % (2.0*np.pi) - np.pi    # wrap to [-pi, pi]
    
    def calc_heading(self, q):
        return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))

def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(node_name='Navigation')

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()