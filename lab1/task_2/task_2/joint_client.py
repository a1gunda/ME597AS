import sys
import rclpy
from rclpy.node import Node
from task_2_interfaces.srv import JointState

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(JointState, 'joint_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = JointState.Request()

    def send_request(self, x, y, z):
        self.req.x = x
        self.req.y = y
        self.req.z = z

        return self.cli.call_async(self.req)


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()
    future = minimal_client.send_request(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    rclpy.spin_until_future_complete(minimal_client, future)
    response = future.result()
    minimal_client.get_logger().info('Result of joint_service: %s' % response.valid)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
