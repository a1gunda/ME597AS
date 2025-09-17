import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class PIDController(Node):
    def __init__(self):
        super().__init__('pid_speed_controller')
        self.subscription = self.create_subscription(
            LaserScan,'/scan',self.listener_callback,10
        )
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.controller_callback)
        self.derr = 0
        self.ierr = 0
        self.dist = 0

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.ranges[0]}')
        self.dist = msg.ranges[0]
    
    def pid_control(self):
        target = 0.35
        dt = 0.1
        err = self.dist - target
        if self.derr*err < 0:
            self.ierr = 0
        else:
            self.ierr = self.ierr + err*dt
        Kp = 0.14
        Kd = 0.02
        Ki = 0.005
        vel = Kp*err + Kd*(err - self.derr)/dt + Ki*self.ierr
        self.derr = err

        return max(min(vel,0.15),-0.15)

    def controller_callback(self):
        vel = self.pid_control()
        pub = Twist()
        pub.linear.x = vel
        self.publisher_.publish(pub)
        self.get_logger().info(f'Publishing: {self.dist}')


def main(args=None):
    rclpy.init(args=args)

    pid_controller = PIDController()

    rclpy.spin(pid_controller)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pid_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()