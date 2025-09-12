import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class MinimalPublisher(Node):
    def __init__(self):
        self.start_time = time.time()
        super().__init__('talker')
        self.publisher_ = self.create_publisher(Float64, 'my_first_topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        self.run_time = time.time() - self.start_time
        msg = Float64()
        msg.data = self.run_time
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
