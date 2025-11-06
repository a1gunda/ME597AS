import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge          # Convert ROS2 <--> cv2 image type
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class RedBallTracker(Node):
    def __init__(self):
        super().__init__('red_ball_tracker')

        # subscriber
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.track_red_ball, 10
        )

        # publisher
        self.publisher_ = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.bridge = CvBridge()        # image converter
        self.trgt_area = 15500          # desired target area
        self.alpha = 0.25               # smoothing factor

        # low-pass filters
        self.filt_cntr_x = None
        self.filt_area = None

        # PID gains
        self.Kp_lin = 0.00004
        self.Ki_lin = 0.0
        self.Kd_lin = 0.00001
        self.Kp_ang = 0.003
        self.Ki_ang = 0.0
        self.Kd_ang = 0.001

        # PID memory
        self.prev_time = self.get_clock().now()
        self.prev_lin_err = 0.0
        self.prev_ang_err = 0.0
        self.int_lin = 0.0
        self.int_ang = 0.0


    def track_red_ball(self, msg):
        # convert to opencv image
        cv2_img = self.bridge.imgmsg_to_cv2(msg)

        # convert to HSV and threshold for red
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 80, 60])
        upper1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        lower2 = np.array([170, 80, 60])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # morphological filtering
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours from mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            self.get_logger().info("Red ball not detected...")
            self.publish_message(0.0, 0.0)
            cv2.imshow('lab3_video', cv2_img)
            cv2.waitKey(1)
            return

        # largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 100:
            self.get_logger().info("Red ball reached...")
            self.publish_message(0.0, 0.0)
            cv2.imshow('lab3_video', cv2_img)
            cv2.waitKey(1)
            return

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2
        fx = cv2_img.shape[1] / 2

        # low pass filtering
        self.filt_cntr_x = self.lpf(self.filt_cntr_x, cx)
        self.filt_area = self.lpf(self.filt_area, area)

        # PID control
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now

        # angular control
        ang_err = fx - self.filt_cntr_x
        ang_vel, self.int_ang = self.pid(
            ang_err, self.prev_ang_err, self.int_ang,
            self.Kp_ang, self.Ki_ang, self.Kd_ang, dt
        )
        self.prev_ang_err = ang_err

        # linear control
        lin_err = self.trgt_area - self.filt_area
        lin_vel, self.int_lin = self.pid(
            lin_err, self.prev_lin_err, self.int_lin,
            self.Kp_lin, self.Ki_lin, self.Kd_lin, dt
        )
        self.prev_lin_err = lin_err

        # clamp speeds
        lin_vel = max(min(lin_vel, 0.2), -0.2)
        ang_vel = max(min(ang_vel, 0.6), -0.6)

        # publish commands
        self.publish_message(lin_vel, ang_vel)

        # visualization
        cv2.circle(cv2_img, (int(cx), int(y+h/2)), 6, (0,255,0), -1)
        cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow('lab3_video', cv2_img)
        cv2.waitKey(1)


    # low pass filter
    def lpf(self, prev_val, new_val):
        if prev_val is None:
            return new_val
        return self.alpha * new_val + (1 - self.alpha) * prev_val


    # PID controller
    def pid(self, err, prev_err, int, Kp, Ki, Kd, dt):
        int += err * dt
        derivative = (err - prev_err) / dt if dt > 0 else 0.0
        output = Kp * err + Ki * int + Kd * derivative
        return output, int
    

    # command publisher
    def publish_message(self, lin_vel, ang_vel):
        twist = Twist()
        twist.linear.x = lin_vel
        twist.angular.z = ang_vel
        self.publisher_.publish(twist)

        self.get_logger().info(
            f'[PID] lin={lin_vel:.4f}, ang={ang_vel:.4f}'
        )


def main(args=None):
    rclpy.init(args=args)

    # instantiate and run
    red_ball_tracker = RedBallTracker()
    rclpy.spin(red_ball_tracker)

    # release image pointer
    red_ball_tracker.cap.release()

    # Destroy the node explicitly
    red_ball_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()