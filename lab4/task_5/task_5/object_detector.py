import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge              # Convert ROS2 <--> cv2 image type
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # initialize subcriber
        self.subscription = self.create_subscription(
            Image, '/video_data', self.detect_image, 10
        )

        # initialize publisher
        self.publisher_ = self.create_publisher(
            BoundingBox2D, '/bbox', 10
        )

        # initialize image converter
        self.bridge = CvBridge()

    def detect_image(self, ros2_img):
        # convert to opencv image
        cv2_img = self.bridge.imgmsg_to_cv2(ros2_img)

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
            self.get_logger().info("No red shapes detected...")
            cv2.imshow('lab3_video', cv2_img)
            cv2.waitKey(1)
            return

        # find only triangles
        triangle_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)

            # must have 3 vertices
            if len(approx) != 3:
                continue

            triangle_contours.append((c, approx, area))

        # detect nothing if no triangles
        if not triangle_contours:
            self.get_logger().info("No triangle found...")
            cv2.imshow('lab3_video', cv2_img)
            cv2.waitKey(1)
            return

        # pick largest triangle
        triangle_contours.sort(key=lambda item: item[2], reverse=True)
        main_triangle, approx_poly, area = triangle_contours[0]

        # bounding rectangle geometry
        x, y, w, h = cv2.boundingRect(main_triangle)
        cx, cy = x + w/2, y + h/2

        # create and publish bounding box
        bbox = BoundingBox2D()
        bbox.center.position.x = float(cx)
        bbox.center.position.y = float(cy)
        bbox.size_x = float(w)
        bbox.size_y = float(h)

        self.publisher_.publish(bbox)
        self.get_logger().info("Publishing bounding box...")

        # visualization
        # cv2.drawContours(cv2_img, [approx_poly], -1, (0,255,0), 3)
        cv2.rectangle(cv2_img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow('lab3_video', cv2_img)
        cv2.waitKey(1)


def imshow(self, log, cv2_img):
    self.get_logger().info(log)
    cv2.imshow('lab3_video', cv2_img)
    cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    # instantiate and run
    object_detector = ObjectDetector()
    rclpy.spin(object_detector)

    # release image pointer
    object_detector.cap.release()

    # Destroy the node explicitly
    object_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
