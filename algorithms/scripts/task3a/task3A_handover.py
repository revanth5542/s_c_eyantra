#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  Task 3A: Inter-Robot Handover Coordination Node.
*  Coordinates synchronization between eBot and UR7e robotic arm during ore package
*  handover beside the arm mount.
*
*****************************************************************************************
'''

import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, String


class Task3AHandover(Node):
    def __init__(self):
        super().__init__(
            'task3A_handover',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Handover state flags
        self.ebot_at_arm = False
        self.package_cleared = False
        self.ebot_returned_home = False
        self.all_ores_sorted = False

        # Subscriptions from eBot and Arm
        self.ebot_at_arm_sub = self.create_subscription(
            Bool, '/strata/ebot_at_arm', self.ebot_at_arm_cb, latched_qos)
        self.pkg_cleared_sub = self.create_subscription(
            Bool, '/strata/package_cleared', self.pkg_cleared_cb, latched_qos)
        self.ebot_home_sub = self.create_subscription(
            Bool, '/strata/ebot_returned_home', self.ebot_home_cb, latched_qos)
        self.ores_sorted_sub = self.create_subscription(
            Bool, '/strata/all_ores_sorted', self.ores_sorted_cb, latched_qos)

        # Re-broadcasters for high-reliability synchronization
        self.ebot_ready_pub = self.create_publisher(Bool, '/strata/ebot_ready_for_pickup', latched_qos)
        self.can_depart_pub = self.create_publisher(Bool, '/strata/ebot_can_depart', latched_qos)

        self.timer = self.create_timer(0.2, self.status_loop)
        self.get_logger().info('Task 3A Inter-Robot Handover Coordinator Initialized.')

    def ebot_at_arm_cb(self, msg: Bool):
        if msg.data and not self.ebot_at_arm:
            self.ebot_at_arm = True
            self.get_logger().info('>>> eBot arrived and stationary at Arm Handover Pose!')
            out = Bool()
            out.data = True
            self.ebot_ready_pub.publish(out)

    def pkg_cleared_cb(self, msg: Bool):
        if msg.data and not self.package_cleared:
            self.package_cleared = True
            self.get_logger().info('>>> Arm confirmed: Ore package successfully lifted off eBot deck!')
            out = Bool()
            out.data = True
            self.can_depart_pub.publish(out)

    def ebot_home_cb(self, msg: Bool):
        if msg.data and not self.ebot_returned_home:
            self.ebot_returned_home = True
            self.get_logger().info('>>> eBot confirmed: Returned to Home pose and at rest.')

    def ores_sorted_cb(self, msg: Bool):
        if msg.data and not self.all_ores_sorted:
            self.all_ores_sorted = True
            self.get_logger().info('>>> Arm confirmed: All 4 conveyor ores picked and sorted.')

    def status_loop(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = Task3AHandover()
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

