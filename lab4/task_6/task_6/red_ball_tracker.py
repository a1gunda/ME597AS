import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class RedBallTracker(Node):
    def __init__(self):
        super().__init__('red_ball_tracker')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.track_red_ball, 10
        )
        self.publisher_   = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        self.bridge       = CvBridge()

        # resize cv2 window
        cv2.namedWindow('task_6', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('task_6', 640, 400)

        # tracking parameters
        self.trgt_area  = 1e4
        self.dband      = 4e2
        self.max_lin    = 0.4
        self.max_ang    = 0.6

        # low-pass filter for d term
        self.filt_d_area = 0.0
        self.alpha       = 0.25

        # PD gains
        self.Kp_lin = 1.2e-4
        self.Kd_lin = 1.8e-4
        self.Kp_ang = 4.2e-4
        self.Kd_ang = 1.2e-4

        # memory for derivative
        self.prev_time     = self.get_clock().now()
        self.prev_area_err = 0
        self.prev_cntr_err = 0


    def track_red_ball(self, msg):
        # convert to cv2 and hsv
        cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv     = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

        # dual range red mask
        mask1 = cv2.inRange(hsv, np.array([  0,80,80]), np.array([ 10,255,255]))
        mask2 = cv2.inRange(hsv, np.array([170,80,80]), np.array([180,255,255]))
        mask  = cv2.bitwise_or(mask1, mask2)

        # clean mask
        kernel = np.ones((7,7), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            self.get_logger().info('No red ball detected...')
            self.publish_message(0, 0)
            cv2.imshow('task_6', cv2_img); cv2.waitKey(1)
            return

        # largest contour
        c       = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)

        # center calculation
        cx, fx  = x + w/2, cv2_img.shape[1] / 2

        # time step
        now = self.get_clock().now()
        dt  = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now

        # -----------------------------
        # linear PD control (distance)
        # -----------------------------
        area_err = self.trgt_area - area
        d_area   = (area_err - self.prev_area_err) / dt if dt > 0 else 0.0
        self.filt_d_area = (1 - self.alpha) * self.filt_d_area + self.alpha * d_area
        self.prev_area_err = area_err

        lin_vel = self.Kp_lin * area_err + self.Kd_lin * self.filt_d_area
        lin_vel = np.clip(lin_vel, -self.max_lin, self.max_lin)

        # deadband to remove jitter
        if abs(area_err) < self.dband or abs(lin_vel) < self.max_lin / 10:
            self.get_logger().info('Target distance reached...')
            lin_vel = 0.0

        # -----------------------------
        # angular PD control (centering)
        # -----------------------------
        cntr_err = fx - cx
        d_cntr   = (cntr_err - self.prev_cntr_err) / dt if dt > 0 else 0.0
        self.prev_cntr_err = cntr_err

        ang_vel = self.Kp_ang * cntr_err + self.Kd_ang * d_cntr
        ang_vel = np.clip(ang_vel, -self.max_ang, self.max_ang)

        # publish motion command
        self.publish_message(lin_vel, ang_vel)

        # visualization
        cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.circle(cv2_img, (int(cx), int(y + h / 2)), 6, (0,255,0), -1)
        cv2.imshow('task_6', cv2_img)
        cv2.waitKey(1)


    def publish_message(self, lin_vel, ang_vel):
        twist = Twist()
        twist.linear.x  = float(lin_vel)
        twist.angular.z = float(ang_vel)
        self.publisher_.publish(twist)
        self.get_logger().info(
            f'[PD] lin={lin_vel:.4f}, ang={ang_vel:.4f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = RedBallTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()