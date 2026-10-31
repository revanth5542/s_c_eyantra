#!/usr/bin/env python3

import rclpy
import sys
import math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32


# 5 official waypoints in base_link frame (meters)[cite: 4, 7]
OFFICIAL_WAYPOINTS = [
    np.array([-0.4085, -0.5379, 0.1967]),  # Waypoint 1[cite: 4, 7]
    np.array([-0.8000, -0.0005, 0.3967]),  # Waypoint 2 (+200mm in Z)
    np.array([-0.7430,  0.5280, 0.1967]),  # Waypoint 3[cite: 4, 7]
    np.array([-0.4097,  0.5280, 0.1967]),  # Waypoint 4[cite: 4, 7]
    np.array([-0.0763,  0.5280, 0.1967]),  # Waypoint 5[cite: 4, 7]
]

# Transition via-point to navigate safely around the base cylinder from start
TRANSITION_VIA = np.array([0.10, -0.45, 0.35])

base_frame = 'base_link'


class arm_waypoints(Node):
    def __init__(self):
        super().__init__(
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # Publishers & Subscriptions[cite: 7]
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        self.control_rate = 0.05  # 20 Hz[cite: 7]
        self.timer = self.create_timer(self.control_rate, self.process_waypoints)

        self.tcp_pose = None
        self.arm_status = None

        # Navigation state
        self.in_transition = True  # Move via TRANSITION_VIA first
        self.wp_idx = 0
        self.is_holding = False
        self.hold_start_sim_time = None
        self.task_completed = False
        self.last_log_sim_time = 0.0

        # Tolerances & tuning[cite: 4, 7]
        self.target_tolerance = 0.022   # Inside 0.03 m for full bonus marks[cite: 4]
        self.hold_duration = 2.4        # Hold duration in sim time >= 2.0 s[cite: 4]
        self.max_linear_speed = 0.08    # Safely under cap_linear_mps = 0.15[cite: 7]
        self.max_rot_speed = 0.15       # Safely under cap_angular_rps = 0.35[cite: 7]

        self.get_logger().info('Arm Waypoint Controller initialized.')

    def tcpposecb(self, data):
        self.tcp_pose = data

    def armstatuscb(self, data):
        self.arm_status = data.data

    def publish_twist(self, vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.x = float(wx)
        msg.twist.angular.y = float(wy)
        msg.twist.angular.z = float(wz)
        self.twist_pub.publish(msg)

    def compute_orientation_velocity(self):
        '''
        Aligns the tool axis (column 2 of R) with vector b = [0, 0, -1] (downwards)[cite: 7].
        Uses the exact cross-product method provided in the task boilerplate[cite: 7].
        '''
        q = self.tcp_pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # Column 2 of rotation matrix (tool longitudinal axis in base_link)
        a_x = 2.0 * (qx * qz + qy * qw)
        a_y = 2.0 * (qy * qz - qx * qw)
        a_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        a = np.array([a_x, a_y, a_z])

        b = np.array([0.0, 0.0, -1.0])  # Desired downward pointing axis[cite: 7]

        c = np.cross(a, b)
        c_norm = np.linalg.norm(c)
        dot_ab = float(np.dot(a, b))

        if c_norm > 1e-4:
            angle = math.atan2(c_norm, dot_ab)
            rot_axis = c / c_norm
            e = rot_axis * angle
            kp_rot = 0.8
            omega = kp_rot * e
            w_norm = np.linalg.norm(omega)
            if w_norm > self.max_rot_speed:
                omega = omega * (self.max_rot_speed / w_norm)
            return omega[0], omega[1], omega[2]
        return 0.0, 0.0, 0.0

    def process_waypoints(self):
        if self.tcp_pose is None or self.arm_status is None:
            return

        if self.task_completed:
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        now_sim_s = self.get_clock().now().nanoseconds / 1e9

        cur_p = np.array([
            self.tcp_pose.pose.position.x,
            self.tcp_pose.pose.position.y,
            self.tcp_pose.pose.position.z
        ])

        # Step A: Handle initial transition around the base cylinder
        if self.in_transition:
            diff = TRANSITION_VIA - cur_p
            dist = float(np.linalg.norm(diff))
            wx, wy, wz = self.compute_orientation_velocity()

            if dist <= 0.06:
                self.in_transition = False
                self.get_logger().info('Transition clear of base. Tracking official Waypoint 1.')
                return

            speed = min(self.max_linear_speed, max(0.02, 0.5 * dist))
            vel = (diff / dist) * speed
            self.publish_twist(vel[0], vel[1], vel[2], wx, wy, wz)
            return

        # Step B: Track official waypoints 1 through 5[cite: 4, 7]
        target_p = OFFICIAL_WAYPOINTS[self.wp_idx]
        diff = target_p - cur_p
        dist = float(np.linalg.norm(diff))

        # Check waypoint arrival condition[cite: 4]
        if dist <= self.target_tolerance:
            if not self.is_holding:
                self.is_holding = True
                self.hold_start_sim_time = now_sim_s
                self.get_logger().info(
                    f'Reached Waypoint {self.wp_idx + 1} (dist={dist:.4f}m <= {self.target_tolerance}m). Holding...'
                )

            # Hold stationary by commanding zero velocity on every tick[cite: 4, 7]
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            elapsed_sim = now_sim_s - self.hold_start_sim_time
            if elapsed_sim >= self.hold_duration:
                self.get_logger().info(
                    f'Waypoint {self.wp_idx + 1} hold completed ({elapsed_sim:.2f}s sim time).'
                )
                self.wp_idx += 1
                self.is_holding = False

                if self.wp_idx >= len(OFFICIAL_WAYPOINTS):
                    self.task_completed = True
                    self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    self.get_logger().info('All 5 waypoints visited and held successfully! Task 1B complete.')
            return

        # Moving towards current official waypoint[cite: 4, 7]
        self.is_holding = False
        speed = min(self.max_linear_speed, max(0.02, 0.6 * dist))
        vel = (diff / dist) * speed
        wx, wy, wz = self.compute_orientation_velocity()

        self.publish_twist(vel[0], vel[1], vel[2], wx, wy, wz)

        if now_sim_s - self.last_log_sim_time > 1.0:
            self.last_log_sim_time = now_sim_s
            self.get_logger().info(
                f'WP {self.wp_idx + 1} | TCP: [{cur_p[0]:.3f}, {cur_p[1]:.3f}, {cur_p[2]:.3f}] | Remaining: {dist:.4f}m | Speed: {speed:.3f}m/s'
            )


def main():
    rclpy.init(args=sys.argv)
    node = arm_waypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
