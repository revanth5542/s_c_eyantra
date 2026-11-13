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

        # State machine variables
        self.tcp_pos = None
        self.arm_status = None
        self.force_mag = 0.0
        self.ebot_ready = False

        self.conveyor_targets = []
        self.conveyor_sorted_count = 0
        self.total_conveyor_ores = 4
        self.phase = 'INIT'
        self.state = 'IDLE'

        self.target_ore_id = None
        self.target_ore_type = None
        self.target_pos = None
        self.approach_pos = None
        self.drop_pos = None

        self.state_timer = 0
        self.magnet_future = None

        # 20 Hz control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Task 3B Manipulation Node Initialized.')

    def tcpposecb(self, msg: PoseStamped):
        p = msg.pose.position
        self.tcp_pos = np.array([p.x, p.y, p.z])

    def armstatuscb(self, msg: Int32):
        self.arm_status = msg.data

    def forcecb(self, msg: Float32):
        self.force_mag = msg.data

    def ebot_ready_cb(self, msg: Bool):
        if msg.data:
            self.ebot_ready = True
            self.get_logger().info('Manipulation: eBot ready for ore package pickup.')

    def set_magnet(self, enable: bool):
        req = SetBool.Request()
        req.data = enable
        self.magnet_future = self.magnet_client.call_async(req)

    def publish_twist(self, linear, angular=None):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = base_frame

        max_linear_speed = 0.23
        max_angular_speed = 0.50

        speed = np.linalg.norm(linear)
        if speed > max_linear_speed:
            linear = (linear / speed) * max_linear_speed

        twist.twist.linear.x = float(linear[0])
        twist.twist.linear.y = float(linear[1])
        twist.twist.linear.z = float(linear[2])

        if angular is not None:
            ang_speed = np.linalg.norm(angular)
            if ang_speed > max_angular_speed:
                angular = (angular / ang_speed) * max_angular_speed
            twist.twist.angular.x = float(angular[0])
            twist.twist.angular.y = float(angular[1])
            twist.twist.angular.z = float(angular[2])
        else:
            twist.twist.angular.x = 0.0
            twist.twist.angular.y = 0.0
            twist.twist.angular.z = 0.0

        self.twist_pub.publish(twist)

    def stop_arm(self):
        self.publish_twist(np.zeros(3), np.zeros(3))

    def move_toward(self, target, tol=0.015, kp=2.2):
        if self.tcp_pos is None:
            return False

        error = target - self.tcp_pos
        dist = np.linalg.norm(error)

        if dist < tol:
            self.stop_arm()
            return True

        direction = error / dist
        vel = min(0.23, max(0.04, kp * dist)) * direction
        self.publish_twist(vel)
        return False

    def discover_conveyor_ores(self):
        known_ores = []
        for o_type in ['azurite_ore', 'malachite_ore', 'vanadinite_ore']:
            for idx in [1, 2, 3]:
                cand_id = f"{o_type}_{idx}"
                try:
                    tf = self.tf_buffer.lookup_transform(base_frame, cand_id, rclpy.time.Time())
                    p = tf.transform.translation
                    known_ores.append((cand_id, o_type, np.array([p.x, p.y, p.z])))
                except Exception:
                    pass

        # Sort North to South (highest Y to lowest Y)
        known_ores.sort(key=lambda item: item[2][1], reverse=True)
        return known_ores

    def control_loop(self):
        if self.tcp_pos is None:
            return

        # -------------------------------------------------------------
        # PHASE 1: INITIALIZATION
        # -------------------------------------------------------------
        if self.phase == 'INIT':
            self.set_magnet(False)
            self.state_timer += 1
            if self.state_timer > 20:
                self.phase = 'CONVEYOR_SCAN'
                self.state = 'SCANNING'
                self.state_timer = 0
                self.get_logger().info('Beginning Phase 1: Conveyor Ores Sorting.')

        # -------------------------------------------------------------
        # PHASE 2: CONVEYOR ORES SORTING
        # -------------------------------------------------------------
        elif self.phase == 'CONVEYOR_SCAN':
            if self.state == 'SCANNING':
                ores = self.discover_conveyor_ores()
                if len(ores) >= 4:
                    self.conveyor_targets = ores[:4]
                    self.get_logger().info(
                        f"Found all 4 conveyor ores: {[o[0] for o in self.conveyor_targets]}"
                    )
                    self.phase = 'SORTING'
                    self.state = 'NEXT_ORE'
                else:
                    self.state_timer += 1
                    if self.state_timer % 20 == 0:
                        self.get_logger().info(
                            f"Waiting for perception... currently found {len(ores)} ores."
                        )

        elif self.phase == 'SORTING':
            if self.state == 'NEXT_ORE':
                if len(self.conveyor_targets) == 0:
                    self.get_logger().info('All conveyor ores sorted! Signalling completion.')
                    sorted_msg = Bool()
                    sorted_msg.data = True
                    self.ores_sorted_pub.publish(sorted_msg)
                    self.phase = 'WAIT_FOR_EBOT'
                    self.state = 'WAITING'
                    return

                self.target_ore_id, self.target_ore_type, self.target_pos = self.conveyor_targets.pop(0)

                # Update target with latest TF
                try:
                    tf = self.tf_buffer.lookup_transform(base_frame, self.target_ore_id, rclpy.time.Time())
                    p = tf.transform.translation
                    self.target_pos = np.array([p.x, p.y, p.z])
                except Exception:
                    pass

                self.approach_pos = np.array([self.target_pos[0], self.target_pos[1], CONVEYOR_APPROACH_Z])
                self.drop_pos = CONTAINER_DROP_POSES[self.target_ore_type].copy()

                self.get_logger().info(
                    f"Sorting #{self.conveyor_sorted_count + 1}: {self.target_ore_id} -> {self.target_ore_type} bin"
                )
                self.state = 'MOVE_CONVEYOR_ENTRY'

            elif self.state == 'MOVE_CONVEYOR_ENTRY':
                entry_wp = np.array([CONVEYOR_ENTRY_WP[0], CONVEYOR_ENTRY_WP[1], SAFE_TRANSITION_Z])
                if self.move_toward(entry_wp, tol=0.04):
                    self.state = 'MOVE_ABOVE_ORE'

            elif self.state == 'MOVE_ABOVE_ORE':
                if self.move_toward(self.approach_pos, tol=0.02):
                    self.state = 'DESCEND_TO_ORE'

            elif self.state == 'DESCEND_TO_ORE':
                contact_pos = np.array([self.target_pos[0], self.target_pos[1], 0.040])
                if self.move_toward(contact_pos, tol=0.012, kp=1.6):
                    self.set_magnet(True)
                    self.state = 'MAGNET_ATTACH'
                    self.state_timer = 0

            elif self.state == 'MAGNET_ATTACH':
                self.state_timer += 1
                if self.state_timer >= 12:
                    self.state = 'LIFT_ORE'

            elif self.state == 'LIFT_ORE':
                lift_pos = np.array([self.tcp_pos[0], self.tcp_pos[1], SAFE_TRANSITION_Z])
                if self.move_toward(lift_pos, tol=0.02):
                    self.state = 'MOVE_SWING_WP'

            elif self.state == 'MOVE_SWING_WP':
                swing_wp = np.array([SWING_WAYPOINT[0], SWING_WAYPOINT[1], SAFE_TRANSITION_Z])
                if self.move_toward(swing_wp, tol=0.04):
                    self.state = 'MOVE_ABOVE_BIN'

            elif self.state == 'MOVE_ABOVE_BIN':
                bin_approach = np.array([self.drop_pos[0], self.drop_pos[1], CONTAINER_APPROACH_Z])
                if self.move_toward(bin_approach, tol=0.03):
                    self.state = 'LOWER_INTO_BIN'

            elif self.state == 'LOWER_INTO_BIN':
                bin_release = np.array([self.drop_pos[0], self.drop_pos[1], CONTAINER_RELEASE_Z])
                if self.move_toward(bin_release, tol=0.02):
                    self.set_magnet(False)
                    self.state = 'MAGNET_RELEASE'
                    self.state_timer = 0

            elif self.state == 'MAGNET_RELEASE':
                self.state_timer += 1
                if self.state_timer >= 10:
                    self.conveyor_sorted_count += 1
                    self.state = 'RETRACT_FROM_BIN'

            elif self.state == 'RETRACT_FROM_BIN':
                bin_retreat = np.array([self.drop_pos[0], self.drop_pos[1], SAFE_TRANSITION_Z])
                if self.move_toward(bin_retreat, tol=0.03):
                    self.state = 'MOVE_SWING_RETURN'

            elif self.state == 'MOVE_SWING_RETURN':
                swing_wp = np.array([SWING_WAYPOINT[0], SWING_WAYPOINT[1], SAFE_TRANSITION_Z])
                if self.move_toward(swing_wp, tol=0.04):
                    self.state = 'NEXT_ORE'

        # -------------------------------------------------------------
        # PHASE 3: WAIT FOR EBOT AND PICK ORE PACKAGE
        # -------------------------------------------------------------
        elif self.phase == 'WAIT_FOR_EBOT':
            if self.state == 'WAITING':
                ready_pose = np.array([0.25, -0.15, SAFE_TRANSITION_Z])
                self.move_toward(ready_pose, tol=0.03)

                if self.ebot_ready:
                    self.get_logger().info('eBot arrived! Localizing ore package for pickup.')
                    self.state = 'RESOLVE_PACKAGE_POSE'
                    self.state_timer = 0

            elif self.state == 'RESOLVE_PACKAGE_POSE':
                self.state_timer += 1
                try:
                    tf = self.tf_buffer.lookup_transform(base_frame, 'ore_package', rclpy.time.Time())
                    p = tf.transform.translation
                    self.pkg_target_pos = np.array([p.x, p.y, p.z])
                except Exception:
                    self.pkg_target_pos = np.array([0.6284, -0.0283, 0.2168])

                if self.state_timer >= 15:
                    self.get_logger().info(
                        f"Resolved package pickup target at ({self.pkg_target_pos[0]:.4f}, "
                        f"{self.pkg_target_pos[1]:.4f}, {self.pkg_target_pos[2]:.4f})"
                    )
                    self.phase = 'PICK_PACKAGE'
                    self.state = 'MOVE_CORRIDOR_TO_EBOT'

        elif self.phase == 'PICK_PACKAGE':
            if self.state == 'MOVE_CORRIDOR_TO_EBOT':
                wp = np.array([0.35, -0.10, SAFE_TRANSITION_Z])
                if self.move_toward(wp, tol=0.03):
                    self.state = 'MOVE_ABOVE_PACKAGE'

            elif self.state == 'MOVE_ABOVE_PACKAGE':
                pkg_above = np.array([self.pkg_target_pos[0], self.pkg_target_pos[1], 0.35])
                if self.move_toward(pkg_above, tol=0.02):
                    self.state = 'DESCEND_TO_PACKAGE'

            elif self.state == 'DESCEND_TO_PACKAGE':
                contact_z = PKG_TOP_SURFACE_Z + 0.002
                pkg_contact = np.array([self.pkg_target_pos[0], self.pkg_target_pos[1], contact_z])
                if self.move_toward(pkg_contact, tol=0.008, kp=1.4):
                    self.set_magnet(True)
                    self.state = 'MAGNET_ATTACH_PACKAGE'
                    self.state_timer = 0

            elif self.state == 'MAGNET_ATTACH_PACKAGE':
                self.state_timer += 1
                if self.state_timer >= 15:
                    self.state = 'LIFT_PACKAGE'

            elif self.state == 'LIFT_PACKAGE':
                pkg_lift = np.array([self.tcp_pos[0], self.tcp_pos[1], 0.38])
                if self.move_toward(pkg_lift, tol=0.02):
                    self.get_logger().info('Ore package cleared eBot deck! Notifying eBot to return Home.')
                    cleared_msg = Bool()
                    cleared_msg.data = True
                    self.pkg_cleared_pub.publish(cleared_msg)

                    self.phase = 'DROP_IN_CRUSHER'
                    self.state = 'MOVE_CRUSHER_WP1'

        # -------------------------------------------------------------
        # PHASE 4: TRANSPORT PACKAGE TO CRUSHER HOPPER
        # -------------------------------------------------------------
        elif self.phase == 'DROP_IN_CRUSHER':
            if self.state == 'MOVE_CRUSHER_WP1':
                if self.move_toward(CRUSHER_CORRIDOR_1, tol=0.04):
                    self.state = 'MOVE_CRUSHER_WP2'

            elif self.state == 'MOVE_CRUSHER_WP2':
                if self.move_toward(CRUSHER_CORRIDOR_2, tol=0.04):
                    self.state = 'MOVE_CRUSHER_APPROACH'

            elif self.state == 'MOVE_CRUSHER_APPROACH':
                if self.move_toward(CRUSHER_HOPPER_APPROACH, tol=0.03):
                    self.state = 'LOWER_INTO_CRUSHER'

            elif self.state == 'LOWER_INTO_CRUSHER':
                if self.move_toward(CRUSHER_HOPPER_LOWER, tol=0.02, kp=1.6):
                    self.set_magnet(False)
                    self.state = 'RELEASE_PACKAGE'
                    self.state_timer = 0

            elif self.state == 'RELEASE_PACKAGE':
                self.state_timer += 1
                if self.state_timer >= 12:
                    self.get_logger().info('Ore package dropped into Ore Crusher hopper successfully!')
                    self.state = 'RETRACT_FROM_CRUSHER'

            elif self.state == 'RETRACT_FROM_CRUSHER':
                if self.move_toward(CRUSHER_HOPPER_APPROACH, tol=0.03):
                    self.state = 'RETURN_HOME'

            elif self.state == 'RETURN_HOME':
                if self.move_toward(HOME_POSE, tol=0.03):
                    self.phase = 'COMPLETE'
                    self.state = 'DONE'
                    self.get_logger().info('UR7e Arm Manipulation Task Completed Successfully.')

        elif self.phase == 'COMPLETE':
            self.stop_arm()


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
