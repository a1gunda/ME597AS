#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class Task1(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self):
        super().__init__('task1_node')

        # ros interface
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.__scan_cbk, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.loop)

        # state and sensor storage
        self.scan = None
        self.state = 'FORWARD'

        # geometry and safety parameters
        self.safe_dist = 0.50
        self.front_safe = 0.55
        self.front_door = 0.30

        # nominal speeds
        self.v_nom = 0.35
        self.w_nom = 1.65

        # controller setup
        self.kp = 1.1
        self.kd = 5.0
        self.kh = 1.2
        self.err_prev = 0.0

        self.get_logger().info('Task 1 controller running')

    ## callbacks
    def __scan_cbk(self, msg):
        # store latest lidar scan
        self.scan = msg

    ## main loop
    def loop(self):
        if self.scan is None:
            return

        # process lidar geometry
        geom = self.comp_geometry(self.scan.ranges)
        cmd = Twist()

        # dispatch control based on current state
        handlers = {
            'FORWARD': self.state_forward,
            'ALIGN': self.state_align,
            'FOLLOW': self.state_follow
        }

        handlers[self.state](geom, cmd)
        self.pub_cmd.publish(cmd)

    ## wall following
    def comp_geometry(self, ranges):
        # front clearance (narrow cone)
        df = self.min_valid(ranges[-10:] + ranges[:10])

        # right-side wall geometry
        drf = self.min_valid(ranges[300:320])
        drb = self.min_valid(ranges[260:280])

        return {
            'df': df,
            'dr': 0.5 * (drf + drb),
            'drf': drf,
            'drb': drb,
            'head_err': np.clip(drf - drb, -0.15, 0.15)
        }

    def state_forward(self, g, cmd):
        # move forward until a wall is detected
        if g['df'] < self.front_safe:
            self.state = 'ALIGN'
            self.state_align(g, cmd)
            return

        cmd.linear.x = 1.05 * self.v_nom

    def state_align(self, g, cmd):
        # rotate left to align with the wall
        if g['df'] > self.front_safe + 0.1:
            self.err_prev = 0.0
            self.state = 'FOLLOW'
            self.state_follow(g, cmd)
            return

        cmd.angular.z = 0.9 * self.w_nom

    def state_follow(self, g, cmd):
        # relax front clearance when right side opens (doorway)
        front_thresh = (
            self.front_door if g['dr'] > 0.75 else self.front_safe
        )

        if g['df'] < front_thresh:
            cmd.angular.z = self.w_nom
            self.err_prev = 0.0
            return

        # lateral distance control
        err = np.clip(g['dr'] - self.safe_dist, -0.4, 0.4)
        derr = err - self.err_prev
        self.err_prev = err

        w = -(self.kp * err + self.kd * derr)

        # apply heading correction only when wall is well-defined
        if g['drf'] < 1.0 and g['drb'] < 1.0:
            w -= self.kh * g['head_err']

        w = np.clip(w, -self.w_nom, self.w_nom)

        cmd.linear.x = 0.9 * self.v_nom
        cmd.angular.z = 1.9 * w

    ## helpers
    def min_valid(self, vals):
        # minimum valid lidar return in a cone
        valid = [v for v in vals if not np.isinf(v) and v > 0.18]
        return min(valid) if valid else 10.0

## main
def main(args=None):
    rclpy.init(args=args)
    node = Task1()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # stop robot cleanly on shutdown
        node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

## save map
# ros2 run nav2_map_server map_saver_cli -f "map" --ros-args -p map_subscribe_transient_local:=true -r