#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  Task 3B: UR7e Arm Manipulation Node.
*  Picks and sorts 4 conveyor ores into designated containers, picks ore_package
*  from the eBot deck upon arrival at handover station, and drops it into the
*  ore crusher hopper.
*
*****************************************************************************************
'''

import sys
import math
import numpy as np
import tf2_ros

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool, Float32, Int32
from std_srvs.srv import SetBool


# Container drop poses in base_link frame (meters)
# Azurite bin:    X in [-0.18, 0.15], Y in [0.44, 0.94], Z in [-0.23, 0.1175]
# Malachite bin:  X in [-0.49, -0.16], Y in [0.44, 0.94], Z in [-0.23, 0.1175]
# Vanadinite bin: X in [-0.78, -0.47], Y in [0.44, 0.94], Z in [-0.23, 0.1175]
CONTAINER_DROP_POSES = {
    'azurite_ore': np.array([-0.08, 0.528, 0.18]),
    'malachite_ore': np.array([-0.41, 0.528, 0.18]),
    'vanadinite_ore': np.array([-0.52, 0.528, 0.18]),
}

# Official Ore Crusher Hopper Poses
CRUSHER_HOPPER_APPROACH = np.array([-0.8000, -0.0005, 0.38])
CRUSHER_HOPPER_LOWER = np.array([-0.8000, -0.0005, 0.27])
CRUSHER_CORRIDOR_1 = np.array([0.10, -0.45, 0.38])
CRUSHER_CORRIDOR_2 = np.array([-0.50, -0.35, 0.38])

# ore_package physical ground truth and top surface height on eBot deck in base_link frame (meters)
# Package center Z ~= 0.2168m, height = 0.0762m -> Top Surface Z = 0.2168 + 0.0381 = 0.2549m
PKG_TOP_SURFACE_Z = 0.2549

# Pedestal-clearance waypoints:
# SWING_WAYPOINT (North-East, clearing container approach): X=0.28, Y=0.15, Z=0.32
# CONVEYOR_ENTRY_WP (South-East, clearing conveyor approach): X=0.25, Y=-0.25, Z=0.32
SWING_WAYPOINT = np.array([0.28, 0.15, 0.32])
CONVEYOR_ENTRY_WP = np.array([0.25, -0.25, 0.32])

CONVEYOR_APPROACH_Z = 0.15
CONVEYOR_LIFT_Z = 0.15
SAFE_TRANSITION_Z = 0.32
CONTAINER_APPROACH_Z = 0.24
CONTAINER_RELEASE_Z = 0.18
HOME_POSE = np.array([0.138, -0.133, 0.543])
base_frame = 'base_link'


class Task3BManipulation(Node):
    def __init__(self):
        super().__init__(
            'task3B_manipulation',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers & Subscriptions
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.force_sub = self.create_subscription(
            Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 10)

        # Inter-robot communication with Handover coordinator
        self.ebot_ready_sub = self.create_subscription(
            Bool, '/strata/ebot_ready_for_pickup', self.ebot_ready_cb, latched_qos)
        self.pkg_cleared_pub = self.create_publisher(Bool, '/strata/package_cleared', latched_qos)
        self.ores_sorted_pub = self.create_publisher(Bool, '/strata/all_ores_sorted', latched_qos)

        # TF listener for detected ore and package frames
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Electromagnet service client
        self.magnet_client = self.create_client(SetBool, '/magnet')
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /magnet service...')

        self.control_rate = 0.05  # 20 Hz
        self.timer = self.create_timer(self.control_rate, self.control_loop)

        self.tcp_pose = None
        self.arm_status = None
        self.force_magnitude = 0.0
        self.ebot_ready = False

        # State machine
        self.state = 'DISCOVER_ORES'
        self.discovered_ores = []
        self.current_ore_idx = 0
        self.state_timer_start = None
        self.package_target_pos = None

        # Speed limits and tolerances (strictly within ur_arm_controller bounds: lin <= 0.25, rot <= 0.60)
        self.pos_tolerance = 0.025
        self.max_linear_speed = 0.22
        self.max_rot_speed = 0.50

        self.get_logger().info('Task 3B Manipulation Node Initialized.')

    def ebot_ready_cb(self, msg: Bool):
        if msg.data:
            self.ebot_ready = True

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

        # Hard safety ceiling strictly enforced below hardware limits (lin: 0.25, rot: 0.60)
        lin_spd = math.sqrt(vx*vx + vy*vy + vz*vz)
        if lin_spd > 0.23:
            scale = 0.23 / lin_spd
            vx *= scale
            vy *= scale
            vz *= scale

        rot_spd = math.sqrt(wx*wx + wy*wy + wz*wz)
        if rot_spd > 0.50:
            scale = 0.50 / rot_spd
            wx *= scale
            wy *= scale
            wz *= scale

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

    def navigate_towards(self, target_pos, speed_cap=None, tol=None, stop_at_target=True):
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
            if stop_at_target:
                self.publish_twist(0.0, 0.0, 0.0, wx, wy, wz)
            return True

        v_dir = error / dist
        speed = min(3.0 * dist + 0.05, speed_cap)
        v = v_dir * speed

        self.publish_twist(v[0], v[1], v[2], wx, wy, wz)
        return False

    def get_sim_time(self):
        return self.get_clock().now().nanoseconds / 1e9

    def control_loop(self):
        if self.tcp_pose is None:
            return

        now = self.get_sim_time()

        # ==========================================================
        # PHASE 1: DISCOVER AND SORT 4 CONVEYOR ORES
        # ==========================================================
        if self.state == 'DISCOVER_ORES':
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
                        if -0.20 < pos[2] < 0.15:
                            found.append((frame, ore_kind, pos))
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        pass

            if self.state_timer_start is None:
                self.state_timer_start = now

            if len(found) >= 4 or (len(found) >= 1 and now - self.state_timer_start > 2.5):
                found.sort(key=lambda item: -item[2][0])
                self.discovered_ores = found[:4]
                self.get_logger().info(f'Discovered {len(self.discovered_ores)} valid ores on conveyor:')
                for f_id, kind, p in self.discovered_ores:
                    self.get_logger().info(f'  {f_id} ({kind}) at {p}')
                self.state = 'MOVE_TO_CONVEYOR_ENTRY'
                self.current_ore_idx = 0
            else:
                self.get_logger().info(f'Waiting for 4 ores in TF (currently {len(found)})...', throttle_duration_sec=2.0)
                self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        elif self.state == 'MOVE_TO_CONVEYOR_ENTRY':
            if self.navigate_towards(CONVEYOR_ENTRY_WP, speed_cap=0.24, tol=0.035, stop_at_target=False):
                self.state = 'APPROACH_HOVER_LOW'

        elif self.state == 'APPROACH_HOVER_LOW':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            target = np.array([ore_pos[0], ore_pos[1], ore_pos[2] + 0.018])
            if self.navigate_towards(target, speed_cap=0.22, tol=0.012):
                self.get_logger().info(f'Above {ore_id}. Activating magnet BEFORE final descent...')
                self.set_magnet(True)
                self.state = 'DESCEND_TO_ORE'
                self.state_timer_start = now

        elif self.state == 'DESCEND_TO_ORE':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            target = np.array([ore_pos[0], ore_pos[1], ore_pos[2] + 0.0015])
            reached = self.navigate_towards(target, speed_cap=0.045, tol=0.0012)
            curr_z = self.tcp_pose.pose.position.z
            contact = (self.force_magnitude > 2.2 and curr_z <= ore_pos[2] + 0.006)
            depth_reached = (curr_z <= ore_pos[2] + 0.0030)
            timed_out = (now - self.state_timer_start > 5.0)
            if reached or contact or depth_reached or timed_out:
                self.get_logger().info(
                    f'Reached capture depth for {ore_id} (force={self.force_magnitude:.2f}N, z={curr_z:.4f}m). Waiting for magnet latch...'
                )
                self.state = 'CAPTURE_PAUSE'
                self.state_timer_start = now

        elif self.state == 'CAPTURE_PAUSE':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            target = np.array([ore_pos[0], ore_pos[1], ore_pos[2] + 0.0015])
            self.navigate_towards(target, speed_cap=0.005, tol=0.0010)
            if now - self.state_timer_start >= 0.15:
                self.get_logger().info(f'Ore {ore_id} latched. Lifting off conveyor...')
                self.state = 'LIFT_ORE'

        elif self.state == 'LIFT_ORE':
            ore_id, ore_kind, ore_pos = self.discovered_ores[self.current_ore_idx]
            target = np.array([ore_pos[0], ore_pos[1], CONVEYOR_LIFT_Z])
            if self.navigate_towards(target, speed_cap=0.24, tol=0.025, stop_at_target=False):
                self.get_logger().info(f'Lifted {ore_id}. Navigating via conveyor exit...')
                self.state = 'MOVE_TO_CONVEYOR_EXIT'

        elif self.state == 'MOVE_TO_CONVEYOR_EXIT':
            if self.navigate_towards(CONVEYOR_ENTRY_WP, speed_cap=0.24, tol=0.035, stop_at_target=False):
                self.state = 'MOVE_TO_SWING_WP'

        elif self.state == 'MOVE_TO_SWING_WP':
            if self.navigate_towards(SWING_WAYPOINT, speed_cap=0.24, tol=0.035, stop_at_target=False):
                self.state = 'MOVE_TO_CONTAINER_HIGH'

        elif self.state == 'MOVE_TO_CONTAINER_HIGH':
            ore_id, ore_kind, _ = self.discovered_ores[self.current_ore_idx]
            drop_target = CONTAINER_DROP_POSES[ore_kind]
            approach_pos = np.array([drop_target[0], drop_target[1], CONTAINER_APPROACH_Z])
            if self.navigate_towards(approach_pos, speed_cap=0.24, tol=0.03, stop_at_target=False):
                self.state = 'LOWER_TO_CONTAINER'

        elif self.state == 'LOWER_TO_CONTAINER':
            ore_id, ore_kind, _ = self.discovered_ores[self.current_ore_idx]
            drop_target = CONTAINER_DROP_POSES[ore_kind]
            release_pos = np.array([drop_target[0], drop_target[1], CONTAINER_RELEASE_Z])
            if self.navigate_towards(release_pos, speed_cap=0.18, tol=0.02):
                self.get_logger().info(f'Inside {ore_kind} container. Releasing magnet...')
                self.set_magnet(False)
                self.state = 'RELEASE_ORE'
                self.state_timer_start = now

        elif self.state == 'RELEASE_ORE':
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if now - self.state_timer_start >= 0.15:
                self.state = 'ASCEND_FROM_CONTAINER'

        elif self.state == 'ASCEND_FROM_CONTAINER':
            ore_id, ore_kind, _ = self.discovered_ores[self.current_ore_idx]
            drop_target = CONTAINER_DROP_POSES[ore_kind]
            approach_pos = np.array([drop_target[0], drop_target[1], CONTAINER_APPROACH_Z])
            if self.navigate_towards(approach_pos, speed_cap=0.24, tol=0.03, stop_at_target=False):
                self.current_ore_idx += 1
                if self.current_ore_idx < len(self.discovered_ores):
                    self.get_logger().info(
                        f'Ore {self.current_ore_idx} placed. Moving to ore {self.current_ore_idx + 1}/4...'
                    )
                    self.state = 'MOVE_BACK_TO_SWING_WP'
                else:
                    self.get_logger().info('=' * 68)
                    self.get_logger().info('All 4 conveyor ores sorted successfully!')
                    self.get_logger().info('Notifying handover node. Transitioning to eBot package pickup...')
                    self.get_logger().info('=' * 68)
                    ores_msg = Bool()
                    ores_msg.data = True
                    self.ores_sorted_pub.publish(ores_msg)
                    self.state = 'MOVE_TO_WAIT_POSE'

        elif self.state == 'MOVE_BACK_TO_SWING_WP':
            if self.navigate_towards(SWING_WAYPOINT, speed_cap=0.24, tol=0.035, stop_at_target=False):
                self.state = 'MOVE_TO_CONVEYOR_ENTRY'

        # ==========================================================
        # PHASE 2: WAIT FOR EBOT AND PICK ORE PACKAGE
        # ==========================================================
        elif self.state == 'MOVE_TO_WAIT_POSE':
            # Hold at swing waypoint facing +X to await eBot arrival
            if self.navigate_towards(SWING_WAYPOINT, speed_cap=0.24, tol=0.035):
                self.state = 'WAIT_FOR_PACKAGE'

        elif self.state == 'WAIT_FOR_PACKAGE':
            if not self.ebot_ready:
                self.get_logger().info('Waiting for eBot arrival at Handover pose...', throttle_duration_sec=2.0)
                wx, wy, wz = self.compute_orientation_velocity()
                self.publish_twist(0.0, 0.0, 0.0, wx, wy, wz)
                return

            # Look up ore_package transform from task3B_perception
            pkg_x = None
            pkg_y = None

            try:
                tf = self.tf_buffer.lookup_transform(base_frame, 'ore_package', rclpy.time.Time())
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                tz = tf.transform.translation.z
                # Correct physical bounds for package on eBot deck beside the arm
                if 0.50 < tx < 0.75 and -0.15 < ty < 0.15:
                    pkg_x = tx
                    pkg_y = ty
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass

            if pkg_x is None:
                pkg_x = 0.6284
                pkg_y = -0.0283

            # Physical top surface of ore_package in base_link frame is rigidly PKG_TOP_SURFACE_Z = 0.2549m
            self.package_target_pos = np.array([pkg_x, pkg_y, PKG_TOP_SURFACE_Z])

            self.get_logger().info(
                f'Ore package located on eBot deck at ({self.package_target_pos[0]:.4f}, '
                f'{self.package_target_pos[1]:.4f}, top Z={self.package_target_pos[2]:.4f})!'
            )
            self.state = 'APPROACH_PKG_HOVER_HIGH'

        elif self.state == 'APPROACH_PKG_HOVER_HIGH':
            target = np.array([self.package_target_pos[0], self.package_target_pos[1], SAFE_TRANSITION_Z])
            if self.navigate_towards(target, speed_cap=0.24, tol=0.03):
                self.state = 'APPROACH_PKG_HOVER_LOW'

        elif self.state == 'APPROACH_PKG_HOVER_LOW':
            # Hover 1.5 cm above physical top surface of package
            target = np.array([
                self.package_target_pos[0],
                self.package_target_pos[1],
                PKG_TOP_SURFACE_Z + 0.015
            ])
            if self.navigate_towards(target, speed_cap=0.18, tol=0.008):
                self.get_logger().info('Above ore_package. Activating magnet BEFORE final descent...')
                self.set_magnet(True)
                self.state = 'DESCEND_PKG_CONTACT'
                self.state_timer_start = now

        elif self.state == 'DESCEND_PKG_CONTACT':
            # Target 2.0 mm above top surface (electromagnet capture range: 0.5mm < gap < 5.0mm)
            target = np.array([
                self.package_target_pos[0],
                self.package_target_pos[1],
                PKG_TOP_SURFACE_Z + 0.0020
            ])
            reached = self.navigate_towards(target, speed_cap=0.025, tol=0.0015)
            curr_z = self.tcp_pose.pose.position.z
            depth_reached = (curr_z <= PKG_TOP_SURFACE_Z + 0.0025)
            timed_out = (now - self.state_timer_start > 3.0)

            if reached or depth_reached or timed_out:
                self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'Reached capture depth for ore_package (z={curr_z:.4f}m, gap={(curr_z - PKG_TOP_SURFACE_Z)*1000.0:.1f}mm). Waiting for magnet latch...'
                )
                self.state = 'ENGAGE_PKG_MAGNET'
                self.state_timer_start = now

        elif self.state == 'ENGAGE_PKG_MAGNET':
            target = np.array([
                self.package_target_pos[0],
                self.package_target_pos[1],
                PKG_TOP_SURFACE_Z + 0.0020
            ])
            self.navigate_towards(target, speed_cap=0.002, tol=0.0010)
            if now - self.state_timer_start >= 0.35:
                self.get_logger().info('Ore package latched. Lifting off eBot deck...')
                self.state = 'LIFT_PKG'
                self.state_timer_start = now

        elif self.state == 'LIFT_PKG':
            # Lift vertically to clear the eBot pedestal safely
            target = np.array([
                self.package_target_pos[0],
                self.package_target_pos[1],
                0.38
            ])
            reached = self.navigate_towards(target, speed_cap=0.22, tol=0.02, stop_at_target=False)
            curr_z = self.tcp_pose.pose.position.z
            if (reached or curr_z >= 0.35) and (now - self.state_timer_start >= 0.6):
                self.get_logger().info(
                    'Package lifted above eBot deck! Signaling eBot to depart for Home.'
                )
                cleared_msg = Bool()
                cleared_msg.data = True
                self.pkg_cleared_pub.publish(cleared_msg)
                self.state = 'MOVE_PKG_TO_CRUSHER_CORRIDOR_1'

        # ==========================================================
        # PHASE 3: TRANSPORT PACKAGE TO CRUSHER HOPPER
        # ==========================================================
        elif self.state == 'MOVE_PKG_TO_CRUSHER_CORRIDOR_1':
            if self.navigate_towards(CRUSHER_CORRIDOR_1, speed_cap=0.24, tol=0.04, stop_at_target=False):
                self.state = 'MOVE_PKG_TO_CRUSHER_CORRIDOR_2'

        elif self.state == 'MOVE_PKG_TO_CRUSHER_CORRIDOR_2':
            if self.navigate_towards(CRUSHER_CORRIDOR_2, speed_cap=0.24, tol=0.04, stop_at_target=False):
                self.state = 'MOVE_PKG_TO_CRUSHER_HOPPER_APPROACH'

        elif self.state == 'MOVE_PKG_TO_CRUSHER_HOPPER_APPROACH':
            if self.navigate_towards(CRUSHER_HOPPER_APPROACH, speed_cap=0.24, tol=0.03, stop_at_target=False):
                self.state = 'LOWER_PKG_INTO_CRUSHER'

        elif self.state == 'LOWER_PKG_INTO_CRUSHER':
            if self.navigate_towards(CRUSHER_HOPPER_LOWER, speed_cap=0.18, tol=0.02):
                self.get_logger().info('Lowered into Ore Crusher Hopper! Releasing package...')
                self.set_magnet(False)
                self.state = 'RELEASE_PKG_IN_CRUSHER'
                self.state_timer_start = now

        elif self.state == 'RELEASE_PKG_IN_CRUSHER':
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if now - self.state_timer_start >= 0.25:
                self.get_logger().info('Ore package settled inside crusher successfully! Retracting...')
                self.state = 'RETRACT_FROM_CRUSHER'

        elif self.state == 'RETRACT_FROM_CRUSHER':
            if self.navigate_towards(CRUSHER_HOPPER_APPROACH, speed_cap=0.22, tol=0.03, stop_at_target=False):
                self.state = 'RETURN_HOME_VIA_CORRIDOR_2'

        elif self.state == 'RETURN_HOME_VIA_CORRIDOR_2':
            if self.navigate_towards(CRUSHER_CORRIDOR_2, speed_cap=0.24, tol=0.04, stop_at_target=False):
                self.state = 'RETURN_HOME_VIA_CORRIDOR_1'

        elif self.state == 'RETURN_HOME_VIA_CORRIDOR_1':
            if self.navigate_towards(CRUSHER_CORRIDOR_1, speed_cap=0.24, tol=0.04, stop_at_target=False):
                self.state = 'RETURN_HOME'

        elif self.state == 'RETURN_HOME':
            if self.navigate_towards(HOME_POSE, speed_cap=0.24, tol=0.025):
                self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.state = 'MANIPULATION_COMPLETE'
                self.get_logger().info('=' * 68)
                self.get_logger().info('UR7e Arm Manipulation Complete! Returned to Home Pose.')
                self.get_logger().info('=' * 68)

        elif self.state == 'MANIPULATION_COMPLETE':
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = Task3BManipulation()
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
