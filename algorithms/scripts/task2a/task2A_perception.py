#!/usr/bin/env python3

################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image


##################### TASK CONSTANTS #######################

# Two ores of each type are spawned - six in all[cite: 1, 3]
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# RealSense camera topics[cite: 7]
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# Parent frame[cite: 1, 7]
base_frame = 'base_link'

# Team ID for saving the detection image
TEAM_ID = '3646'

# HSV color boundaries for Gazebo simulation ores[cite: 1]
COLOR_BOUNDS = {
    'azurite_ore': {
        'lower': np.array([95, 100, 50]),
        'upper': np.array([135, 255, 255]),
        'draw_color': (255, 0, 0)
    },
    'malachite_ore': {
        'lower': np.array([35, 70, 50]),
        'upper': np.array([85, 255, 255]),
        'draw_color': (0, 255, 0)
    },
    'vanadinite_ore': {
        'lower': np.array([5, 120, 70]),
        'upper': np.array([25, 255, 255]),
        'draw_color': (0, 140, 255)
    }
}


##################### HELPER FUNCTIONS #######################

def transform_point(x, y, z, transform_msg):
    t = transform_msg.transform.translation
    q = transform_msg.transform.rotation

    q_vec = np.array([q.x, q.y, q.z], dtype=np.float64)
    p_vec = np.array([x, y, z], dtype=np.float64)

    cross1 = np.cross(q_vec, p_vec) + q.w * p_vec
    p_rot = p_vec + 2.0 * np.cross(q_vec, cross1)

    return float(p_rot[0] + t.x), float(p_rot[1] + t.y), float(p_rot[2] + t.z)


def detect_ores(image):
    center_ore_list = []
    ore_type_list = []

    if image is None:
        return center_ore_list, ore_type_list

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    for ore_type in ore_types:
        bounds = COLOR_BOUNDS[ore_type]
        mask = cv2.inRange(hsv, bounds['lower'], bounds['upper'])

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            # Ores on conveyor are typically between 100 and 800 px
            if 100 < area < 800:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    # Restrict detection strictly to the conveyor surface
                    # Exclude hopper on the left (cX < 430) and lower machinery (cY > 420)
                    if 430 < cX < 610 and 180 < cY < 420:
                        valid_candidates.append((area, (cX, cY), c))

        valid_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, (cX, cY), c in valid_candidates[:4]:
            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)
            cv2.drawContours(image, [c], -1, bounds['draw_color'], 2)

    return center_ore_list, ore_type_list


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    def __init__(self):
        super().__init__('ore_tf_publisher')

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.2, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.cam_info = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.optical_frame = 'camera_color_optical_frame'

    def depthimagecb(self, data):
        try:
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                self.depth_image = depth.astype(np.float32) / 1000.0
            else:
                self.depth_image = depth.astype(np.float32)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Color conversion failed: {e}")

    def caminfocb(self, data):
        if self.cam_info is None:
            self.cam_info = data
            self.fx = data.k[0]
            self.cx = data.k[2]
            self.fy = data.k[4]
            self.cy = data.k[5]
            if data.header.frame_id:
                self.optical_frame = data.header.frame_id

    def process_image(self):
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        display_image = self.cv_image.copy()
        centers, types = detect_ores(display_image)

        try:
            tf_to_base = self.tf_buffer.lookup_transform(
                base_frame,
                self.optical_frame,
                rclpy.time.Time()
            )
        except tf2_ros.TransformException:
            return

        h, w = self.depth_image.shape[:2]
        now = self.get_clock().now().to_msg()
        published_ores_count = 0

        for ore_type in ore_types:
            type_centers = [pt for pt, t in zip(centers, types) if t == ore_type]
            type_centers.sort(key=lambda p: (p[1], p[0]))

            for idx, (cX, cY) in enumerate(type_centers):
                ore_id = f"{ore_type}_{idx + 1}"

                y_min = max(0, cY - 4)
                y_max = min(h, cY + 5)
                x_min = max(0, cX - 4)
                x_max = min(w, cX + 5)

                patch = self.depth_image[y_min:y_max, x_min:x_max]
                valid_depths = patch[np.isfinite(patch) & (patch > 0.0)]

                if len(valid_depths) == 0:
                    continue

                z_cam = float(np.median(valid_depths))
                x_cam = (cX - self.cx) * z_cam / self.fx
                y_cam = (cY - self.cy) * z_cam / self.fy

                x_b, y_b, z_b = transform_point(x_cam, y_cam, z_cam, tf_to_base)

                # Ores on the conveyor surface are around z_b = -0.065 m
                # Reject floor/machinery reflections (e.g. z_b < -0.15 m or z_b > 0.10 m)
                if z_b < -0.15 or z_b > 0.10:
                    continue

                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = base_frame
                t.child_frame_id = ore_id
                t.transform.translation.x = x_b
                t.transform.translation.y = y_b
                t.transform.translation.z = z_b
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                self.br.sendTransform(t)
                published_ores_count += 1

                cv2.circle(display_image, (cX, cY), 4, (0, 0, 255), -1)
                cv2.putText(display_image, ore_id, (cX - 35, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(display_image, ore_id, (cX - 35, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Save detection snapshot once all 4 ores are detected
        if published_ores_count == 4:
            cv2.imwrite(f'SC#{TEAM_ID}_task2A_detection.png', display_image)

        cv2.imshow("Ore Detection - Task 2A", display_image)
        cv2.waitKey(1)


##################### MAIN FUNCTION #######################

def main():
    rclpy.init(args=sys.argv)
    ore_tf_node = ore_tf()
    try:
        rclpy.spin(ore_tf_node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        ore_tf_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
