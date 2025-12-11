import os
import cv2                              # OpenCV
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge          # Convert ROS2 <--> cv2 image type
from sensor_msgs.msg import Image       # To use a ROS2 image message


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        # initialize publisher
        self.publisher_ = self.create_publisher(
            Image, '/video_data', 10
        )
        
        # initialize image converter
        self.bridge = CvBridge()

        # capture image from file
        self.home_path = os.path.expanduser('~')
        self.image_path = "Documents/ros2_ws/src/task_5/resource/lab3_video.avi"
        self.cap = cv2.VideoCapture(os.path.join(self.home_path, self.image_path))
        
        # publish at fps rate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timer = self.create_timer(1/fps, self.publish_image)

    def publish_image(self):
        # read image from capture
        success, cv2_img = self.cap.read()
        if success: # successful read
            # convert to ros2 image
            ros2_img = self.bridge.cv2_to_imgmsg(cv2_img, encoding='bgr8')
            ros2_img.header.frame_id = "camera_optical_frame"
            ros2_img.header.stamp = self.get_clock().now().to_msg()

            # publish converted image
            self.publisher_.publish(ros2_img)
            self.get_logger().info('Publishing image...')
        else:       # unsuccessful read
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.get_logger().info('Looping video...')


def main(args=None):
    rclpy.init(args=args)

    # instantiate and run
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)

    # release image pointer
    image_publisher.cap.release()

    # Destroy the node explicitly
    image_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
