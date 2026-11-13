#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  Task 3B: eBot Navigation & Path Tracking Controller.
*  Smooth pure-pursuit trajectory follower on /cmd_vel with acceleration limiting
*  to prevent package slippage, tight waypoint tolerances, and terminal in-place
*  yaw orientation alignment.
*
*****************************************************************************************
'''

import sys
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


# Topics
path_topic = '/ebot_path'
odom_topic = '/odom'
cmd_topic = '/cmd_vel'

# Velocity bounds
MAX_LINEAR_SPEED = 0.45     # Under 0.50 m/s limit
MAX_ANGULAR_SPEED = 1.5     # rad/s
WAYPOINT_TOLERANCE = 0.15   # Intermediate waypoint tolerance (metres) - tight tracking around obstacles
GOAL_TOLERANCE = 0.12       # Final step pose tolerance (metres)


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Task3BNavigation(Node):
    def __init__(self):
        super().__init__(
            'task3B_navigation',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, path_qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        self.control_rate = 0.05  # 20 Hz
        self.timer = self.create_timer(self.control_rate, self.process_navigation)

        self.route = None
        self.odom = None
        self.current_wp_idx = 0
        self.step_completed = False
        self.aligning = False
        self.prev_v_lin = 0.0

        self.get_logger().info('Task 3B Navigation Controller Initialized.')

    def pathcb(self, data: Path):
        if len(data.poses) < 2:
            return

        new_route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]

        if self.route is None or self.route != new_route or self.step_completed:
            self.route = new_route
            self.current_wp_idx = 0
            self.step_completed = False
            self.aligning = False
            self.prev_v_lin = 0.0
            self.get_logger().info(
                f'Received new leg path with {len(self.route)} waypoints: '
                f'start={self.route[0]} -> goal={self.route[-1]}.'
            )

    def odomcb(self, data: Odometry):
        self.odom = data

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        cmd = Twist()
        cmd.linear.x = float(max(-0.1, min(MAX_LINEAR_SPEED, linear_x)))
        cmd.angular.z = float(max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angular_z)))
        self.cmd_pub.publish(cmd)

    def process_navigation(self):
        if self.route is None or self.odom is None:
            return

        if self.step_completed:
            self.publish_cmd(0.0, 0.0)
            return

        # 1. Current pose from /odom
        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # 2. Advance intermediate waypoints if within tolerance
        while self.current_wp_idx < len(self.route) - 1:
            tx, ty = self.route[self.current_wp_idx]
            dist_to_wp = math.hypot(tx - rx, ty - ry)
            if dist_to_wp <= WAYPOINT_TOLERANCE:
                self.current_wp_idx += 1
            else:
                break

        is_last_wp = (self.current_wp_idx == len(self.route) - 1)
        tx, ty = self.route[self.current_wp_idx]
        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)

        # 3. Final waypoint arrival & in-place yaw alignment check
        if is_last_wp:
            if dist <= 0.04 or self.aligning:
                self.aligning = True

                # Determine target yaw
                if math.hypot(tx, ty) < 0.5:
                    target_yaw = 0.0
                else:
                    err_0 = abs(wrap_to_pi(0.0 - yaw))
                    err_pi = abs(wrap_to_pi(math.pi - yaw))
                    target_yaw = 0.0 if err_0 <= err_pi else math.pi

                yaw_err = wrap_to_pi(target_yaw - yaw)

                if abs(yaw_err) > 0.03:
                    v_ang = max(-0.85, min(0.85, 2.5 * yaw_err))
                    if abs(v_ang) < 0.20:
                        v_ang = 0.20 if yaw_err > 0 else -0.20
                    self.publish_cmd(0.0, v_ang)
                else:
                    self.publish_cmd(0.0, 0.0)
                    self.step_completed = True
                    self.aligning = False
                    self.prev_v_lin = 0.0
                    self.get_logger().info(
                        f'Leg goal reached & aligned: pos=({rx:.2f}, {ry:.2f}) dist={dist:.3f}m, '
                        f'yaw={yaw:.2f} (target={target_yaw:.2f}, err={yaw_err:.3f} rad). Holding standstill.'
                    )
                return

        # 4. Heading error to active waypoint
        target_heading = math.atan2(dy, dx)
        heading_err = wrap_to_pi(target_heading - yaw)

        # 5. Smooth drive command (rotate in-place when heading error > 0.25 rad to eliminate turn drift)
        if abs(heading_err) > 0.25:
            v_lin = 0.0
            v_ang = max(-0.95, min(0.95, 2.5 * heading_err))
            if abs(v_ang) < 0.20:
                v_ang = 0.20 if heading_err > 0 else -0.20
        else:
            if is_last_wp:
                v_lin = min(0.35, 0.7 * dist + 0.05) * max(0.0, math.cos(heading_err))
            else:
                v_lin = min(MAX_LINEAR_SPEED, 0.9 * dist + 0.12) * max(0.0, math.cos(heading_err))
            v_lin = max(0.10, v_lin)
            v_ang = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, 2.4 * heading_err))

        # Acceleration rate limiter to prevent ore package from slipping on deck
        max_lin_accel = 0.075
        v_lin = max(self.prev_v_lin - max_lin_accel, min(self.prev_v_lin + max_lin_accel, v_lin))
        self.prev_v_lin = v_lin

        self.publish_cmd(v_lin, v_ang)


def main(args=None):
    rclpy.init(args=args)
    node = Task3BNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
