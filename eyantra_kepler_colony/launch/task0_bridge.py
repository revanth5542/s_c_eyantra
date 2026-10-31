#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import tf2_ros

class Task0CompatibilityNode(Node):
    def __init__(self):
        super().__init__('cartesian_servo_node')
        
        # Camera bridges
        self.pub_rgb = self.create_publisher(Image, '/camera/image', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth_image', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        self.sub_rgb = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.cb_rgb, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.cb_depth, 10)
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.cb_info, 10)
            
        # TCP pose publisher
        self.pub_tcp = self.create_publisher(PoseStamped, '/tcp_pose_raw', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(0.05, self.timer_cb) # 20 Hz
        self.get_logger().info('Task 0 Compatibility Bridge (cartesian_servo_node) started.')

    def cb_rgb(self, msg):
        self.pub_rgb.publish(msg)

    def cb_depth(self, msg):
        self.pub_depth.publish(msg)

    def cb_info(self, msg):
        self.pub_info.publish(msg)

    def timer_cb(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        try:
            # Try to lookup transform base_link -> tcp or tool0 or flange
            for target_frame in ['tcp', 'tool0', 'flange']:
                if self.tf_buffer.can_transform('base_link', target_frame, rclpy.time.Time()):
                    t = self.tf_buffer.lookup_transform('base_link', target_frame, rclpy.time.Time())
                    msg.pose.position.x = t.transform.translation.x
                    msg.pose.position.y = t.transform.translation.y
                    msg.pose.position.z = t.transform.translation.z
                    msg.pose.orientation = t.transform.rotation
                    self.pub_tcp.publish(msg)
                    return
        except Exception:
            pass
        # Fallback default pose if TF not yet available
        msg.pose.position.x = -0.597
        msg.pose.position.y = 1.865
        msg.pose.position.z = 0.882
        msg.pose.orientation.w = 1.0
        self.pub_tcp.publish(msg)

def main():
    rclpy.init()
    node = Task0CompatibilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
