import rclpy
from rclpy.node import Node
from task_2_interfaces.srv import JointState

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(JointState, 'joint_service', self.joint_service_callback)

    def joint_service_callback(self, request, response):
        self.get_logger().info('Incoming request\nx: %f y: %f z: %f' % (request.x, request.y, request.z))
        response.valid = (request.x + request.y + request.z) >= 0
        self.get_logger().info('Sent response\nvalid: %s' % response.valid)

        return response


def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
