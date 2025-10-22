#!/usr/bin/env python3

import sys
import os
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32

from navigation_astar_f24 import *

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
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

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
        self.mp = MapProcessor('sync_classroom_map')
        kr = self.mp.rect_kernel(5,1)
        self.mp.inflate_map(kr,True)
        self.mp.get_graph_from_map()
        # plot
        fig, ax = plt.subplots(dpi=100)
        plt.imshow(self.mp.inf_map_img_array)
        plt.colorbar()
        plt.show()

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Star path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path = Path()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        
        mp.map_graph.root = f"{start_pose.pose.position.y},{start_pose.pose.position.x}"
        mp.map_graph.end = f"{end_pose.pose.position.y},{end_pose.pose.position.x}"

        as_maze = AStar(mp.map_graph)
        start = time.time()
        as_maze.solve(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])
        end = time.time()
        print('Elapsed Time: %.3f'%(end - start))
        
        path_as,dist_as = as_maze.reconstruct_path(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])
        path_arr_as = mp.draw_path(path_as)

        fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi=300, sharex=True, sharey=True)
        ax[0].imshow(path_arr_as)
        ax[0].set_title('Path A*')
        plt.show()
        print(dist_as)

        path.poses.append(start_pose)
        for pose in path_as:
            temp = PoseStamped()
            y,x = map(int,pose.split(","))
            temp.pose.position.x = x
            temp.pose.position.y = y

            path.poses.append(temp)
        path.poses.append(end_pose)
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        return path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path            Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose    PoseStamped object containing the current vehicle position.
        @return idx             Position in the path pointing to the next goal pose to follow.
        """
        idx = 0
        min_dist = float('inf')
        vx = vehicle_pose.pose.position.x
        vy = vehicle_pose.pose.position.y
        vhead = self.get_heading(vehicle_pose.pose.orientation)

        for i,pose in enumerate(path.poses):
            # extract coordinates
            x = pose.pose.position.x
            y = pose.pose.position.y

            # distance and heading
            dist = np.sqrt((x - vx)**2 + (y - vy)**2)
            head = np.atan2(y - vy, x - vx)
            head_err = head - vhead

            # only check ahead
            if abs(head_err) < (np.pi/2):
                if dist < min_dist:
                    min_dist = dist
                    idx = i
        
        return idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        speed = 0.0
        heading = 0.0
        # TODO: IMPLEMENT PATH FOLLOWER
        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
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
            # Call the spin_once to handle callbacks
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks without blocking

            # 1. Create the path to follow
            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            # 2. Loop through the path and move the robot
            idx = self.get_path_idx(path, self.ttbot_pose)
            current_goal = path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)

            self.rate.sleep()
            # Sleep for the rate to control loop timing

    def get_heading(self, quat):
        """! Converts the quaternion of the robot to the heading.
        @param quat         orientation quaternion
        @return heading     heading angle
        """
        sin = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cos = 1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2)

        return np.atan2(sin, cos)


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
