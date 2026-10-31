#!/usr/bin/env python3

import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data

# Topics
path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

# Velocity bounds specified in boilerplate[cite: 14]
max_linear_mps = 0.45    # Under hardware cap 0.5 m/s[cite: 14]
max_angular_rps = 1.0    # Limit 1.0 rad/s[cite: 14]
WAYPOINT_TOLERANCE = 0.38  # Within official 0.5 m scoring radius[cite: 12]


def wrap_to_pi(angle):
    '''Wraps an angle to the range [-pi, pi][cite: 14].'''
    return math.atan2(math.sin(angle), math.cos(angle))


class ebot_nav(Node):
    def __init__(self):
        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # QoS for latched /ebot_path: Transient Local + Reliable[cite: 12, 14]
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, path_qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        # Control cycle timer (20 Hz)[cite: 14]
        control_rate = 0.05
        self.timer = self.create_timer(control_rate, self.process_navigation)

        self.route = None
        self.odom = None
        self.scan = None
        self.current_wp_idx = 0
        self.task_completed = False
        self.last_log_time = 0.0

        self.get_logger().info('eBot Navigation Controller initialized. Waiting for latched route...')

    def pathcb(self, data: Path):
        # Store route once upon arrival[cite: 14]
        if self.route is None and len(data.poses) > 0:
            self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]
            self.get_logger().info(
                f'Received latched route with {len(self.route)} waypoints in {data.header.frame_id} frame.'
            )

    def odomcb(self, data: Odometry):
        self.odom = data

    def scancb(self, data: LaserScan):
        self.scan = data

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        cmd = Twist()
        cmd.linear.x = float(max(-0.1, min(max_linear_mps, linear_x)))
        cmd.angular.z = float(max(-max_angular_rps, min(max_angular_rps, angular_z)))
        self.cmd_pub.publish(cmd)

    def process_navigation(self):
        # Return early until all sensor streams have arrived[cite: 14]
        if self.route is None or self.odom is None or self.scan is None:
            return

        if self.task_completed:
            self.publish_cmd(0.0, 0.0)
            return

        now_sim_s = self.get_clock().now().nanoseconds / 1e9

        # 1. Current pose and heading from /odom[cite: 14]
        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # 2. Distance to active waypoint[cite: 14]
        tx, ty = self.route[self.current_wp_idx]
        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)

        is_last_wp = (self.current_wp_idx == len(self.route) - 1)
        arrival_threshold = 0.25 if is_last_wp else WAYPOINT_TOLERANCE

        # 3. Check waypoint passage[cite: 12]
        if dist <= arrival_threshold:
            self.get_logger().info(
                f'Reached Waypoint {self.current_wp_idx + 1}/10 (dist={dist:.3f}m <= {arrival_threshold}m).'
            )
            if is_last_wp:
                self.task_completed = True
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info('Final waypoint reached! eBot stopped.')
                return
            else:
                self.current_wp_idx += 1
                tx, ty = self.route[self.current_wp_idx]
                dx = tx - rx
                dy = ty - ry
                dist = math.hypot(dx, dy)

        # 4. Heading error to target waypoint[cite: 14]
        target_heading = math.atan2(dy, dx)
        heading_err = wrap_to_pi(target_heading - yaw)

        # 5. Front-cone obstacle evaluation via LiDAR[cite: 14, 15, 19]
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        ranges = self.scan.ranges

        min_front_dist = float('inf')
        left_clearance = float('inf')
        right_clearance = float('inf')

        for i, r in enumerate(ranges):
            # Ignore inf and out-of-range beams[cite: 14, 19]
            if not math.isfinite(r) or r < self.scan.range_min or r > self.scan.range_max:
                continue

            beam_angle = angle_min + i * angle_inc

            # Central front sector ([-30 deg, +30 deg])
            if abs(beam_angle) < 0.52:
                if r < min_front_dist:
                    min_front_dist = r

            # Left flank ([+15 deg, +75 deg])
            if 0.26 < beam_angle < 1.31:
                if r < left_clearance:
                    left_clearance = r

            # Right flank ([-75 deg, -15 deg])
            if -1.31 < beam_angle < -0.26:
                if r < right_clearance:
                    right_clearance = r

        # 6. Reactive navigation blending[cite: 14, 16]
        if min_front_dist < 1.0:
            # Obstacle directly ahead; steer toward the side with greater room[cite: 14]
            dodge_dir = 1.0 if left_clearance >= right_clearance else -1.0
            urgency = (1.0 - min_front_dist) / 0.8  # Increases as robot closes in
            avoid_bias = dodge_dir * min(1.2, 1.4 * urgency)

            steer_angle = wrap_to_pi(0.35 * heading_err + avoid_bias)
            speed = max(0.12, 0.30 * (1.0 - 0.5 * urgency))
        else:
            # Clear corridor: track waypoint heading directly[cite: 14]
            steer_angle = heading_err
            speed = max_linear_mps * max(0.15, math.cos(steer_angle))

        # Decelerate smoothly on approach to the final waypoint stop[cite: 12]
        if is_last_wp:
            speed = min(speed, max(0.10, 0.6 * dist))

        # Close loop on heading using proportional gain[cite: 14, 19]
        kp_yaw = 2.0
        ang_z = kp_yaw * steer_angle

        self.publish_cmd(speed, ang_z)

        # Periodic status logging
        if now_sim_s - self.last_log_time > 1.0:
            self.last_log_time = now_sim_s
            self.get_logger().info(
                f'Tracking WP {self.current_wp_idx + 1}/10 | Pos: [{rx:.2f}, {ry:.2f}] | WP Dist: {dist:.3f}m | Front Obs: {min_front_dist:.2f}m'
            )


def main():
    rclpy.init(args=sys.argv)
    node = ebot_nav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
