#!/usr/bin/env python3

import rclpy
import sys
import math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32, Float32
from std_srvs.srv import SetBool
import tf2_ros


# Target drop bins in base_link frame (meters)
# Container drop poses in base_link frame (meters)
# Evaluator bounding boxes:
# Azurite:    x in [-0.18, 0.15], y in [0.44, 0.94], z in [-0.23, 0.12]
# Malachite:  x in [-0.49, -0.16], y in [0.44, 0.94], z in [-0.23, 0.12]
# Vanadinite: x in [-0.78, -0.47], y in [0.44, 0.94], z in [-0.23, 0.12]
# Using Y = 0.528m (the official waypoint Y from Task 1B) places drops well inside the bins
# while keeping total horizontal reach comfortably under 0.77m (max reach limit is 0.817m).
CONTAINER_DROP_POSES = {
    'azurite_ore': np.array([-0.08, 0.528, 0.15]),
    'malachite_ore': np.array([-0.41, 0.528, 0.15]),
    'vanadinite_ore': np.array([-0.55, 0.528, 0.15]),
}

# Continuous circular corridor around +X to keep shoulder rotation smooth and monotonic:
# Conveyor side (-77 deg) <-> Mid bridge (0 deg) <-> Rover side (+98 deg)
CONVEYOR_GATEWAY = np.array([0.10, -0.45, 0.38])
TRANSITION_WAYPOINT = np.array([0.35, 0.0, 0.38])
ROVER_GATEWAY = np.array([-0.08, 0.528, 0.30])

SAFE_TRANSITION_Z = 0.38
CONTAINER_APPROACH_Z = 0.30
CONTAINER_RELEASE_Z = 0.15
HOME_POSE = np.array([0.138, -0.133, 0.543])
base_frame = 'base_link'


class ArmManipulation(Node):
    def __init__(self):
        super().__init__(
            'task2A_manipulation',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # Publishers & Subscriptions
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.force_sub = self.create_subscription(Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 10)

        # TF listener to look up detected ore frames
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Service client for the electromagnet
        self.magnet_client = self.create_client(SetBool, '/magnet')
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /magnet service...')

        self.control_rate = 0.05  # 20 Hz
        self.timer = self.create_timer(self.control_rate, self.control_loop)

        self.tcp_pose = None
        self.arm_status = None
        self.force_magnitude = 0.0

        # State machine
        self.state = 'DISCOVER_ORES'
        self.discovered_ores = []
        self.current_ore_idx = 0
        self.state_timer_start = None

        # Speed limits and tolerances
        self.pos_tolerance = 0.022
        self.max_linear_speed = 0.08
        self.max_rot_speed = 0.15

        self.get_logger().info('Task 2A Manipulation Node Initialized.')

    def tcpposecb(self, msg):
        self.tcp_pose = msg

    def forcecb(self, msg):
        self.force_magnitude = float(msg.data)

    def armstatuscb(self, msg):
        self.arm_status = msg.data
        if self.arm_status != 0:
            self.get_logger().warn(f'Arm status code: {self.arm_status}', throttle_duration_sec=2.0)

    def set_magnet(self, enable: bool):
        req = SetBool.Request()
        req.data = enable
        self.magnet_client.call_async(req)
        self.get_logger().info(f'Set magnet -> {enable}')

    def publish_twist(self, vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0):
        if not rclpy.ok():
            return
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
        '''Aligns tool vertical axis pointing downwards towards ground.'''
        q = self.tcp_pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        a_x = 2.0 * (qx * qz + qy * qw)
        a_y = 2.0 * (qy * qz - qx * qw)
        a_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        a = np.array([a_x, a_y, a_z])

        b = np.array([0.0, 0.0, -1.0])

        c = np.cross(a, b)
        c_norm = np.linalg.norm(c)
        dot_ab = float(np.dot(a, b))

        if c_norm > 1e-4:
            angle = math.atan2(c_norm, dot_ab)
            axis = c / c_norm
            omega = axis * (1.5 * angle)
            spd = np.linalg.norm(omega)
            if spd > self.max_rot_speed:
                omega = omega * (self.max_rot_speed / spd)
            return omega[0], omega[1], omega[2]
        return 0.0, 0.0, 0.0

    def navigate_towards(self, target_pos, speed_cap=None, tol=None):
        if speed_cap is None:
            speed_cap = self.max_linear_speed
        if tol is None:
            tol = self.pos_tolerance

        curr_pos = np.array([
            self.tcp_pose.pose.position.x,
            self.tcp_pose.pose.position.y,
            self.tcp_pose.pose.position.z
        ])

        error = target_pos - curr_pos
        dist = np.linalg.norm(error)

        wx, wy, wz = self.compute_orientation_velocity()

        if dist < tol:
            self.publish_twist(0.0, 0.0, 0.0, wx, wy, wz)
            return True

        v_dir = error / dist
        speed = min(0.6 * dist + 0.015, speed_cap)
        v = v_dir * speed

        self.publish_twist(v[0], v[1], v[2], wx, wy, wz)
        return False

    def get_sim_time(self):
        return self.get_clock().now().nanoseconds / 1e9

    def control_loop(self):
        if self.tcp_pose is None:
            return

        now = self.get_sim_time()

        if self.state == 'DISCOVER_ORES':
            # Check for transforms of ores published by task2A_perception
            found = []
            for ore_kind in ['azurite_ore', 'malachite_ore', 'vanadinite_ore']:
                for idx in range(1, 5):
                    frame = f'{ore_kind}_{idx}'
                    try:
                        tf = self.tf_buffer.lookup_transform(base_frame, frame, rclpy.time.Time())
                        pos = np.array([
                            tf.transform.translation.x,
                            tf.transform.translation.y,
                            tf.transform.translation.z
                        ])
                        # Only take ores on conveyor belt (z_b around -0.065 m)
                        if -0.15 < pos[2] < 0.10:
                            found.append((frame, ore_kind, pos))
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        pass

            if len(found) >= 4:
                # Sort ores: pick from nearest to rover bed (+X) to farthest (-X)
                # Matches expected output order (azurite -> malachite -> vanadinite)
                found.sort(key=lambda item: -item[2][0])
                self.discovered_ores = found[:4]
                self.get_logger().info(f'Discovered {len(self.discovered_ores)} valid ores on conveyor:')
                for f_id, kind, p in self.discovered_ores:
                    self.get_logger().info(f'  {f_id} ({kind}) at {p}')
                self.state = 'APPROACH_HOVER_HIGH'
                self.current_ore_idx = 0
            else:
                self.get_logger().info(f'Waiting for 4 ores in TF (currently {len(found)})...', throttle_duration_sec=2.0)
                self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        elif self.state == 'APPROACH_HOVER_HIGH':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            # First move directly above the ore at safe transition height
            target = np.array([ore_pos[0], ore_pos[1], SAFE_TRANSITION_Z])
            if self.navigate_towards(target, speed_cap=0.08, tol=0.03):
                self.state = 'APPROACH_HOVER_LOW'

        elif self.state == 'APPROACH_HOVER_LOW':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            # Lower vertically to 5 cm above ore top surface
            target = np.array([ore_pos[0], ore_pos[1], ore_pos[2] + 0.05])
            if self.navigate_towards(target, speed_cap=0.06, tol=0.015):
                self.get_logger().info(f'Above {ore_id}. Activating magnet BEFORE final descent...')
                self.set_magnet(True)
                self.state = 'DESCEND_TO_ORE'
                self.state_timer_start = now

        elif self.state == 'DESCEND_TO_ORE':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            # Lower gently to 1.5 mm above ore top surface (capture range is 5.0 mm, min_gap is 0.5 mm)
            target = np.array([ore_pos[0], ore_pos[1], ore_pos[2] + 0.0015])
            reached = self.navigate_towards(target, speed_cap=0.015, tol=0.0012)
            # Contact force check: if force rises above 2.0 N, physical touch is detected
            contact = (self.force_magnitude > 2.0)
            # Allow up to 8 seconds for the descent
            if reached or contact or (now - self.state_timer_start > 8.0):
                self.get_logger().info(f'Reached capture depth for {ore_id} (force={self.force_magnitude:.1f}N). Waiting for magnet latch...')
                self.state = 'CAPTURE_PAUSE'
                self.state_timer_start = now

        elif self.state == 'CAPTURE_PAUSE':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            # Hold stationary at 1.5 mm to ensure magnet joint forms solidly
            target = np.array([ore_pos[0], ore_pos[1], ore_pos[2] + 0.0015])
            self.navigate_towards(target, speed_cap=0.005, tol=0.0010)
            if now - self.state_timer_start >= 1.5:
                self.get_logger().info(f'Ore {ore_id} latched. Lifting off conveyor...')
                self.state = 'LIFT_ORE'

        elif self.state == 'LIFT_ORE':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            # Lift straight up to safe clearance height
            target = np.array([ore_pos[0], ore_pos[1], SAFE_TRANSITION_Z])
            if self.navigate_towards(target, speed_cap=0.06, tol=0.03):
                self.get_logger().info(f'Lifted {ore_id}. Navigating via circular corridor to rover...')
                self.state = 'TRANSITION_TO_CONVEYOR_GATEWAY'

        elif self.state == 'TRANSITION_TO_CONVEYOR_GATEWAY':
            # Step 1 of corridor: align shoulder with conveyor side (-77 deg)
            if self.navigate_towards(CONVEYOR_GATEWAY, speed_cap=0.08, tol=0.04):
                self.state = 'TRANSITION_TO_MID_WAYPOINT'

        elif self.state == 'TRANSITION_TO_MID_WAYPOINT':
            # Step 2 of corridor: transit behind arm mount (0 deg)
            if self.navigate_towards(TRANSITION_WAYPOINT, speed_cap=0.08, tol=0.04):
                self.state = 'TRANSITION_TO_ROVER_GATEWAY'

        elif self.state == 'TRANSITION_TO_ROVER_GATEWAY':
            # Step 3 of corridor: align shoulder with rover side (+98 deg)
            if self.navigate_towards(ROVER_GATEWAY, speed_cap=0.08, tol=0.04):
                ore_id, ore_kind, _ = self.discovered_ores[self.current_ore_idx]
                self.get_logger().info(f'Rover reached. Moving above {ore_kind} bin...')
                self.state = 'MOVE_ABOVE_BIN'

        elif self.state == 'MOVE_ABOVE_BIN':
            ore_id, ore_kind, _ = self.discovered_ores[self.current_ore_idx]
            bin_pos = CONTAINER_DROP_POSES[ore_kind]
            # Slide along rover bins at clearance height
            target = np.array([bin_pos[0], bin_pos[1], CONTAINER_APPROACH_Z])
            if self.navigate_towards(target, speed_cap=0.08, tol=0.03):
                self.state = 'LOWER_INTO_BIN'

        elif self.state == 'LOWER_INTO_BIN':
            ore_id, ore_kind, _ = self.discovered_ores[self.current_ore_idx]
            bin_pos = CONTAINER_DROP_POSES[ore_kind]
            # Lower into the container bin before releasing
            target = np.array([bin_pos[0], bin_pos[1], CONTAINER_RELEASE_Z])
            if self.navigate_towards(target, speed_cap=0.04, tol=0.025):
                self.get_logger().info(f'Inside {ore_kind} container. Releasing magnet...')
                self.set_magnet(False)
                self.state = 'WAIT_SETTLE'
                self.state_timer_start = now

        elif self.state == 'WAIT_SETTLE':
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if now - self.state_timer_start > 1.2:
                self.state = 'RAISE_FROM_BIN'

        elif self.state == 'RAISE_FROM_BIN':
            ore_id, ore_kind, _ = self.discovered_ores[self.current_ore_idx]
            bin_pos = CONTAINER_DROP_POSES[ore_kind]
            # Lift back up to clearance height above bin
            target = np.array([bin_pos[0], bin_pos[1], CONTAINER_APPROACH_Z])
            if self.navigate_towards(target, speed_cap=0.06, tol=0.03):
                self.state = 'RETURN_TO_ROVER_GATEWAY'

        elif self.state == 'RETURN_TO_ROVER_GATEWAY':
            # Slide back to rover gateway along Y = 0.528
            if self.navigate_towards(ROVER_GATEWAY, speed_cap=0.08, tol=0.04):
                self.current_ore_idx += 1
                if self.current_ore_idx < len(self.discovered_ores):
                    self.get_logger().info(f'Moving back via circular corridor for next ore ({self.current_ore_idx + 1}/4)...')
                    self.state = 'RETURN_TO_MID_WAYPOINT'
                else:
                    self.get_logger().info('All 4 ores have been sorted. Parking arm...')
                    self.state = 'PARK_TRANSITION'

        elif self.state == 'RETURN_TO_MID_WAYPOINT':
            # Return Step 1: transit behind arm mount (0 deg)
            if self.navigate_towards(TRANSITION_WAYPOINT, speed_cap=0.08, tol=0.04):
                self.state = 'RETURN_TO_CONVEYOR_GATEWAY'

        elif self.state == 'RETURN_TO_CONVEYOR_GATEWAY':
            # Return Step 2: align shoulder with conveyor side (-77 deg)
            if self.navigate_towards(CONVEYOR_GATEWAY, speed_cap=0.08, tol=0.04):
                self.state = 'APPROACH_HOVER_HIGH'

        elif self.state == 'PARK_TRANSITION':
            # Return via transition waypoint to avoid cutting through central base cylinder
            if self.navigate_towards(TRANSITION_WAYPOINT, speed_cap=0.08, tol=0.04):
                self.state = 'PARK_HOME'

        elif self.state == 'PARK_HOME':
            # Move arm back to rest pose
            if self.navigate_towards(HOME_POSE, speed_cap=0.08, tol=0.03):
                self.state = 'FINISHED'

        elif self.state == 'FINISHED':
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.get_logger().info(
                'Task 2A complete: All ores sorted into containers. You can now press q in the evaluator.',
                throttle_duration_sec=5.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArmManipulation()
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
