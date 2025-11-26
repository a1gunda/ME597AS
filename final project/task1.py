#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import time

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class Task1(Node):
    """
    Environment mapping task.
    """

    def __init__(self):
        super().__init__('task1')

        self.timer = self.create_timer(0.1, self.timer_cb)

        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.read_map, 10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.read_odom, 10
        )
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.read_scan, 10
        )

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.map = None
        self.pose = None
        self.scan = None

        self.goal = None

        # PID terms
        self.Kp = 1.5
        self.Ki = 0.01
        self.Kd = 0.2

        self.prev_error = 0.0
        self.int_error = 0.0

        # Stuck detection
        self.last_movement_time = time.time()
        self.last_pose = None

        # Time limit: 5 minutes
        self.start_time = self.get_clock().now()

        self.map_saved = False

    # ---------------- Callbacks ----------------

    def read_map(self, msg):
        self.map = msg

    def read_odom(self, msg):
        self.pose = msg.pose.pose

    def read_scan(self, msg):
        self.scan = msg

    # ---------------- Main Loop ----------------

    def timer_cb(self):
        if self.map is None or self.pose is None or self.scan is None:
            return

        # Stop after 5 minutes
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed > 300:
            self.finish_mapping()
            return

        # Stuck detection
        if self.is_stuck():
            self.recovery_spin()
            self.goal = None
            return

        # Get new goal if needed
        if self.goal is None or self.reached_goal():
            self.goal = self.find_best_frontier()
            if self.goal is None:
                self.finish_mapping()
                return

        cmd = self.compute_command()
        self.cmd_publisher.publish(cmd)

    # ---------------- Exploration Logic ----------------

    def reached_goal(self, tol=0.25):
        x = self.pose.position.x
        y = self.pose.position.y
        gx, gy = self.goal
        return math.hypot(gx - x, gy - y) < tol

    def find_best_frontier(self):
        data = self.map.data
        w = self.map.info.width
        h = self.map.info.height
        res = self.map.info.resolution
        origin = self.map.info.origin.position

        rx = self.pose.position.x
        ry = self.pose.position.y
        rmx = int((rx - origin.x) / res)
        rmy = int((ry - origin.y) / res)

        frontiers = []

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                i = y * w + x
                if data[i] != 0:
                    continue

                neighbors = [
                    (x + 1, y), (x - 1, y),
                    (x, y + 1), (x, y - 1)
                ]

                if any(data[ny * w + nx] == -1 for nx, ny in neighbors):
                    size = self.count_frontier_size(x, y, data, w, h)
                    dist = math.hypot(x - rmx, y - rmy)
                    score = dist - size  # prioritize large + nearby
                    frontiers.append((score, x, y))

        if not frontiers:
            return None

        frontiers.sort()
        _, gx, gy = frontiers[0]

        wx = gx * res + origin.x
        wy = gy * res + origin.y

        self.get_logger().info(f"➡ Frontier goal: ({wx:.2f}, {wy:.2f})")
        return (wx, wy)

    def count_frontier_size(self, x, y, data, w, h):
        count = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if data[ny * w + nx] == -1:
                        count += 1
        return count

    # ---------------- Motion + PID Control ----------------

    def compute_command(self):
        cmd = Twist()

        x = self.pose.position.x
        y = self.pose.position.y
        yaw = quat_to_yaw(self.pose.orientation)
        gx, gy = self.goal

        angle_to_goal = math.atan2(gy - y, gx - x)
        err = math.atan2(math.sin(angle_to_goal - yaw), math.cos(angle_to_goal - yaw))

        # PID for angular velocity
        self.int_error += err
        self.int_error = max(min(self.int_error, 1.0), -1.0)
        d_error = err - self.prev_error
        self.prev_error = err

        ang = self.Kp * err + self.Ki * self.int_error + self.Kd * d_error

        dist = math.hypot(gx - x, gy - y)

        # Obstacle avoidance
        blocked = self.is_obstacle_ahead()

        if blocked:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif abs(err) > 0.3:
            cmd.linear.x = 0.0
            cmd.angular.z = ang
        else:
            cmd.linear.x = min(0.25, dist * 0.4)
            cmd.angular.z = ang

        return cmd

    def is_obstacle_ahead(self):
        ranges = self.scan.ranges
        n = len(ranges)
        span = int(n * 20 / 180.0)
        c = n // 2
        front = [r for r in ranges[c-span:c+span] if not math.isinf(r)]
        return front and min(front) < 0.25

    # ---------------- Recovery Behavior ----------------

    def is_stuck(self):
        if self.last_pose is None:
            self.last_pose = self.pose.position
            return False

        dx = self.pose.position.x - self.last_pose.x
        dy = self.pose.position.y - self.last_pose.y
        dist = math.hypot(dx, dy)

        if dist > 0.05:
            self.last_pose = self.pose.position
            self.last_movement_time = time.time()
            return False

        return (time.time() - self.last_movement_time) > 3.0

    def recovery_spin(self):
        self.get_logger().warn("⚠ Robot stuck - performing recovery spin")
        t = Twist()
        t.angular.z = 0.7
        self.cmd_publisher.publish(t)

    # ---------------- Map Saving ----------------

    def finish_mapping(self):
        if not self.map_saved:
            self.save_map_files()
            self.map_saved = True

        self.stop_robot()

    def save_map_files(self):
        self.get_logger().info("💾 Saving map to map_task1.pgm + map_task1.yaml")

        w = self.map.info.width
        h = self.map.info.height
        res = self.map.info.resolution
        origin = self.map.info.origin.position

        # Save PGM
        with open("map_task1.pgm", "w") as f:
            f.write("P2\n")
            f.write(f"{w} {h}\n255\n")
            for y in range(h):
                for x in range(w):
                    val = self.map.data[y * w + x]
                    if val == -1:
                        f.write("205 ")
                    elif val == 0:
                        f.write("254 ")
                    else:
                        f.write("0 ")
                f.write("\n")

        # Save YAML
        with open("map_task1.yaml", "w") as f:
            f.write("image: map_task1.pgm\n")
            f.write(f"resolution: {res}\n")
            f.write(f"origin: [{origin.x}, {origin.y}, 0.0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.196\n")

    def stop_robot(self):
        t = Twist()
        self.cmd_publisher.publish(t)


# ---------------- Main ----------------

def main(args=None):
    rclpy.init(args=args)
    node = Task1()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()