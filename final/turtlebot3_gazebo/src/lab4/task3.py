#!/usr/bin/env python3
import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped

## task 3 node
class Task3(Node):
    """
    Environment localization and navigation task.
    You can also inherit from Task 2 node if most of the code is duplicated
    """
    def __init__(self):
        super().__init__('task3_node')
        # self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # ros interface
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.__scan_cbk, 10)
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.__img_cbk, 10)
        self.sub_pose = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__pose_cbk, 10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_red = self.create_publisher(Point, '/red_pos', 10)
        self.pub_blue = self.create_publisher(Point, '/blue_pos', 10)
        self.pub_green = self.create_publisher(Point, '/green_pos', 10)

        self.timer = self.create_timer(0.1, self.loop)
        self.bridge = CvBridge()

        # resize cv2 window
        cv2.namedWindow('task3_detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('task3_detection', 640, 400)

        # state and sensor storage
        self.finished = False
        self.scan = None
        self.pose = None
        self.yaw = 0.0
        self.state = 'FORWARD'

        # wall-follow geometry and speeds
        self.safe_dist = 0.55
        self.front_safe = 0.60
        self.front_door = 0.30

        self.v_nom = 0.35
        self.w_nom = 1.65

        # wall-follow control (PD + heading)
        self.kp = 1.1
        self.kd = 5.0
        self.kh = 1.2
        self.err_prev = 0.0

        # vision detection settings (tune lightly if needed)
        self.min_area_localize = 1.6e4     # "big enough" gating
        self.center_px_thresh = 18         # pixel deadband for centering
        self.center_kp = 0.0045            # px -> rad/s
        self.center_wmax = 1.2
        self.last_img = None

        # distance/bearing model (heuristic, but consistent)
        self.hfov_deg = 62.0               # approximate turtlebot camera HFOV
        self.dist_k = 6.5                  # distance ~ dist_k / sqrt(area)
        self.dist_min = 0.25
        self.dist_max = 2.25

        # detection bookkeeping
        self.done = {'red': False, 'blue': False, 'green': False}
        self.cooldown = 1.0
        self.last_publish_time = -1e9

        self.target = None                # {'color': str, 'area': float, 'cx': float, 'w': int, 'stamp': float}
        self.target_hold = 0              # small stability counter

        self.get_logger().info('Task 3 running (wall-follow + vision interrupt)')

    ## callbacks
    def __scan_cbk(self, msg):
        self.scan = msg

    def __pose_cbk(self, msg):
        self.pose = msg.pose.pose
        self.yaw = self.calc_heading(self.pose.orientation)

    def __img_cbk(self, msg):
        if all(self.done.values()):
            self.get_logger().info('All balls localized — stop')
            self.finished = True
            return

        cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        det = self.detect_best_ball(cv2_img)
        self.last_img = cv2_img.copy()

        t = self.now_s()
        if det is None:
            self.target = None
            self.target_hold = 0
            return

        det['stamp'] = t

        # keep a tiny amount of temporal stability to reduce flicker
        if self.target is not None and det['color'] == self.target['color']:
            if abs(det['cx'] - self.target['cx']) < 35 and abs(det['area'] - self.target['area']) / max(det['area'], 1.0) < 0.35:
                self.target_hold = min(self.target_hold + 1, 5)
            else:
                self.target_hold = 0
        else:
            self.target_hold = 0

        self.target = det
        self.get_logger().debug(
            f'Vision: candidate {det["color"]}, area={det["area"]:.0f}'
        )

    ## main loop
    def loop(self):
        if self.scan is None:
            return
        
        # stop robot once all balls found
        if self.finished:
            self.pub_cmd.publish(Twist())
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        cmd = Twist()

        # vision interrupt (only when not cooling down)
        if self.state == 'FOLLOW' and self.should_track_ball():
            self.get_logger().info('Ball detected — switching to TRACK mode')
            self.state = 'TRACK'

        if self.state == 'TRACK':
            self.state_track(cmd)
        else:
            g = self.comp_geometry(self.scan.ranges)
            handlers = {
                'FORWARD': self.state_forward,
                'ALIGN': self.state_align,
                'FOLLOW': self.state_follow
            }
            handlers[self.state](g, cmd)

        self.pub_cmd.publish(cmd)

    ## wall following
    def state_forward(self, g, cmd):
        self.get_logger().debug('Wall following: FORWARD')

        if g['df'] < self.front_safe:
            self.get_logger().info('Front wall detected — switching to ALIGN')
            self.state = 'ALIGN'
            self.state_align(g, cmd)
            return
        cmd.linear.x = 1.05 * self.v_nom

    def state_align(self, g, cmd):
        self.get_logger().debug('Wall following: ALIGN')
        
        if g['df'] > self.front_safe + 0.1:
            self.get_logger().info('Alignment complete — switching to FOLLOW')
            self.err_prev = 0.0
            self.state = 'FOLLOW'
            self.state_follow(g, cmd)
            return
        cmd.angular.z = 0.9 * self.w_nom

    def state_follow(self, g, cmd):
        self.get_logger().debug('Wall following: FOLLOW')
        self.get_logger().debug(
            f'FOLLOW geom: df={g["df"]:.2f}, dr={g["dr"]:.2f}'
        )
        front_thresh = self.front_door if g['dr'] > 0.75 else self.front_safe

        if g['df'] < front_thresh:
            cmd.angular.z = self.w_nom
            self.err_prev = 0.0
            return

        err = np.clip(g['dr'] - self.safe_dist, -0.4, 0.4)
        derr = err - self.err_prev
        self.err_prev = err

        w = -(self.kp * err + self.kd * derr)

        if g['drf'] < 1.0 and g['drb'] < 1.0:
            w -= self.kh * g['head_err']

        w = np.clip(w, -self.w_nom, self.w_nom)

        cmd.linear.x = 0.9 * self.v_nom
        cmd.angular.z = 1.9 * w

    def state_track(self, cmd):
        self.get_logger().debug('TRACK mode active')

        # safe default: no forward motion while tracking
        cmd.linear.x = 0.0

        # if we don't have a stable target, just go back to wall following
        if self.target is None or self.target_hold < 1:
            self.state = 'FOLLOW'
            cmd.angular.z = 0.0
            return

        # basic gating: don't waste time on tiny detections
        if self.target['area'] < self.min_area_localize:
            self.get_logger().debug(
                f'Vision: {self.target["color"]} too small (area={self.target["area"]:.0f}), ignoring'
            )
            self.state = 'FOLLOW'
            cmd.angular.z = 0.0
            return
        
        # visualize only when detection is large enough to localize
        self.show_detection()

        # if pose isn't ready, don't publish a fake estimate
        if self.pose is None:
            cmd.angular.z = 0.0
            self.state = 'FOLLOW'
            return

        # rotate to center in frame
        fx = 0.5 * float(self.target['w'])
        cntr_err = fx - float(self.target['cx'])
        w = np.clip(self.center_kp * cntr_err, -self.center_wmax, self.center_wmax)
        self.get_logger().debug(
            f'TRACK centering: color={self.target["color"]}, px_err={cntr_err:.1f}'
        )
        cmd.angular.z = float(w)

        # once centered, estimate and publish
        if abs(cntr_err) <= self.center_px_thresh:
            if self.now_s() - self.last_publish_time < self.cooldown:
                self.state = 'FOLLOW'
                cmd.angular.z = 0.0
                return

            self.get_logger().info(
                f'Localizing {self.target["color"]} ball (area={self.target["area"]:.0f})'
            )
            p = self.estimate_ball_point(self.target['cx'], self.target['w'], self.target['area'])
            if p is not None:
                self.publish_ball(self.target['color'], p)
                self.done[self.target['color']] = True
                self.last_publish_time = self.now_s()

                self.err_prev = 0.0
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

                try:
                    cv2.destroyWindow('task3_detection')
                    cv2.waitKey(1)
                except cv2.error:
                    pass

            self.target = None
            self.target_hold = 0
            self.get_logger().info('Returning to wall following')
            self.state = 'FOLLOW'
            cmd.angular.z = 0.0

    ## geometry / LIDAR
    def comp_geometry(self, ranges):
        df = self.min_valid(list(ranges[-10:]) + list(ranges[:10]))
        drf = self.min_valid(list(ranges[300:320]))
        drb = self.min_valid(list(ranges[260:280]))

        return {
            'df': df,
            'dr': 0.5 * (drf + drb),
            'drf': drf,
            'drb': drb,
            'head_err': np.clip(drf - drb, -0.15, 0.15)
        }

    def min_valid(self, vals):
        valid = [v for v in vals if not np.isinf(v) and v > 0.18]
        return min(valid) if valid else 10.0

    ## vision
    def detect_best_ball(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, W = bgr.shape[:2]

        candidates = []

        if not self.done['red']:
            mask = self.mask_red(hsv)
            det = self.largest_blob(mask)
            if det is not None:
                det['color'] = 'red'
                det['w'] = W
                candidates.append(det)

        if not self.done['blue']:
            mask = self.mask_blue(hsv)
            det = self.largest_blob(mask)
            if det is not None:
                det['color'] = 'blue'
                det['w'] = W
                candidates.append(det)

        if not self.done['green']:
            mask = self.mask_green(hsv)
            det = self.largest_blob(mask)
            if det is not None:
                det['color'] = 'green'
                det['w'] = W
                candidates.append(det)

        if not candidates:
            return None

        # pick the most confident (largest area) among colors not yet done
        return max(candidates, key=lambda d: d['area'])

    def clean_mask(self, mask):
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def largest_blob(self, mask):
        mask = self.clean_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < 200.0:
            return None

        x, y, w, h = cv2.boundingRect(c)
        cx = x + 0.5 * w

        # quick circularity-ish filter (keeps it simple)
        ar = w / max(h, 1)
        if ar < 0.55 or ar > 1.85:
            return None

        return {'area': area, 'cx': float(cx)}

    def mask_red(self, hsv):
        # red wraps hue, so dual-band
        m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        return cv2.bitwise_or(m1, m2)

    def mask_blue(self, hsv):
        # blue band (tweak if needed)
        mask = cv2.inRange(hsv, np.array([100, 90, 70]), np.array([130, 255, 255]))
        return mask

    def mask_green(self, hsv):
        # green band (tweak if needed)
        mask = cv2.inRange(hsv, np.array([40, 90, 70]), np.array([85, 255, 255]))
        return mask

    def show_detection(self):
        if self.last_img is None or self.target is None:
            return

        img = self.last_img.copy()
        cx = int(self.target['cx'])
        h, w = img.shape[:2]

        # draw center line and centroid
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.circle(img, (cx, h // 2), 6, (0, 255, 0), -1)

        txt = f'{self.target["color"]} | area={int(self.target["area"])}'
        cv2.putText(
            img, txt, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.imshow('task3_detection', img)
        cv2.waitKey(1)

    ## localization + publishing
    def should_track_ball(self):
        if self.target is None:
            return False
        if self.target['color'] is None:
            return False
        if self.done.get(self.target['color'], False):
            return False
        if self.target['area'] < self.min_area_localize:
            return False
        if self.now_s() - self.last_publish_time < self.cooldown:
            return False
        return True

    def estimate_ball_point(self, cx, img_w, area):
        rx = self.pose.position.x
        ry = self.pose.position.y
        yaw = self.yaw

        theta = self.pixel_to_bearing(cx, img_w)
        dist = self.area_to_distance(area)

        if not np.isfinite(dist):
            return None

        p = Point()
        p.x = float(rx + dist * np.cos(yaw + theta))
        p.y = float(ry + dist * np.sin(yaw + theta))
        p.z = 0.0
        return p

    def pixel_to_bearing(self, cx, img_w):
        # map pixel offset to angle using approximate HFOV
        fx = 0.5 * float(img_w)
        off = (float(cx) - fx) / max(fx, 1.0)
        hfov = np.deg2rad(self.hfov_deg)
        return float(off * 0.5 * hfov)

    def area_to_distance(self, area):
        # distance heuristic: d ~ k / sqrt(area)
        d = float(self.dist_k / np.sqrt(max(area, 1.0)))
        return float(np.clip(d, self.dist_min, self.dist_max))

    def publish_ball(self, color, p):
        if color == 'red':
            self.pub_red.publish(p)
        elif color == 'blue':
            self.pub_blue.publish(p)
        elif color == 'green':
            self.pub_green.publish(p)
        else:
            return

        self.get_logger().warn(f'Published {color} ball: ({p.x:.3f}, {p.y:.3f})')

    ## helpers
    def now_s(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def calc_heading(self, q):
        return float(np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        ))

## main
def main(args=None):
    rclpy.init(args=args)
    task3 = Task3()

    try:
        rclpy.spin(task3)
    except KeyboardInterrupt:
        pass
    finally:
        # stop robot cleanly on shutdown
        task3.pub_cmd.publish(Twist())
        task3.destroy_node()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
