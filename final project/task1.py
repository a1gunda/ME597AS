#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Import other python packages that you think necessary
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist

class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.read_map, 10
        )
        self.map = None

        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.read_odom, 10
        )
        self.pose = None

        self.cmd_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

    def timer_cb(self):
        self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function

    # Define function(s) that complete the (automatic) mapping task
    def read_map(self, map):
        self.map.origin = map.info.origin
        self.map.height = map.info.height
        self.map.width  = map.info.width
        self.map.data   = map.data

        self.get_logger().info(f'Map origin: {self.map.origin}')

    def read_odom(self, odom):
        self.pose = odom.pose

        self.get_logger().info(f'Odom pose: {self.pose}')


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