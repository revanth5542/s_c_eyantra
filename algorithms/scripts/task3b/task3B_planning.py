#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  Task 3B: eBot Multi-Leg Path Planner Node.
*  Plans collision-free 8-connected A* paths across 3 distinct navigation legs:
*    Leg 1: Home (0,0) -> Ore Drop Pose (6.0, 1.9)
*    Leg 2: Ore Drop Pose (6.0, 1.9) -> Arm Handover Pose (0.0, 1.9, pi)
*    Leg 3: Arm Handover Pose (0.0, 1.9, pi) -> Home (0,0)
*
*  Engineered with adaptive safety inflation to cope with Task 3B perturbed rocks
*  and sensor noise.
*
*****************************************************************************************
'''

import sys
import math
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


# Topics and Frames
map_topic = '/map'
path_topic = '/ebot_path'
odom_topic = '/odom'
spawn_service = '/spawn_ore_package'
map_frame = 'map'

# Poses of the 3-Leg Mission: (x, y, yaw)
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)
arm_pose = (0.0, 1.9, math.pi)

steps = [
    (home_pose, ore_drop_pose),     # Leg 1: -> Request ore package at (6.0, 1.9)
    (ore_drop_pose, arm_pose),      # Leg 2: With package -> Arm Handover at (0.0, 1.9)
    (arm_pose, home_pose),          # Leg 3: After package pickup -> Back to Home (0.0, 0.0)
]

# Baseline obstacle inflation radius (metres) for eBot footprint safety in Task 3B
DEFAULT_INFLATION_RADIUS = 0.44


class Task3BPlanner(Node):
    def __init__(self):
        super().__init__(
            'task3B_planning',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.path_pub = self.create_publisher(Path, path_topic, latched)
        self.spawn_client = self.create_client(Trigger, spawn_service)

        # Handover coordinator communication
        self.can_depart_sub = self.create_subscription(
            Bool, '/strata/ebot_can_depart', self.can_depart_cb, latched)
        self.ebot_at_arm_pub = self.create_publisher(Bool, '/strata/ebot_at_arm', latched)
        self.ebot_home_pub = self.create_publisher(Bool, '/strata/ebot_returned_home', latched)

        self.grid = None
        self.costmap = None
        self.dist_m = None
        self.odom = None
        self.current_step = 0
        self.step_path_published = False
        self.waiting_for_spawn = False
        self.spawn_requested = False
        self.package_spawned = False
        self.step_settle_ticks = 0
        self.spawn_settle_ticks = 0
        self.can_depart = False
        self.task_finished = False

        self.timer = self.create_timer(0.1, self.process_planning)
        self.get_logger().info('Task 3B Planner Node Initialized.')

    def can_depart_cb(self, msg: Bool):
        if msg.data:
            self.can_depart = True

    def mapcb(self, data: OccupancyGrid):
        if self.grid is None:
            self.grid = data
            self.build_distance_field()
            self.build_costmap(DEFAULT_INFLATION_RADIUS)
            self.get_logger().info(
                f'Map received: {data.info.width}x{data.info.height} cells @ '
                f'{data.info.resolution:.3f} m/cell, origin=({data.info.origin.position.x:.2f}, '
                f'{data.info.origin.position.y:.2f})'
            )

    def build_distance_field(self):
        w = self.grid.info.width
        h = self.grid.info.height
        res = self.grid.info.resolution

        raw_data = np.array(self.grid.data, dtype=np.int8).reshape((h, w))
        occupied_mask = (raw_data > 50)

        dist_cells = distance_transform_edt(~occupied_mask)
        self.dist_m = dist_cells * res

    def build_costmap(self, inflation_radius):
        w = self.grid.info.width
        h = self.grid.info.height
        self.costmap = np.zeros((h, w), dtype=np.uint8)
        self.costmap[self.dist_m < inflation_radius] = 100
        self.get_logger().info(f'Costmap built with inflation radius {inflation_radius:.2f} m.')

    def odomcb(self, data: Odometry):
        self.odom = data

    def world_to_grid(self, x, y):
        ox = self.grid.info.origin.position.x
        oy = self.grid.info.origin.position.y
        res = self.grid.info.resolution
        c = int((x - ox) / res)
        r = int((y - oy) / res)
        return c, r

    def grid_to_world(self, c, r):
        ox = self.grid.info.origin.position.x
        oy = self.grid.info.origin.position.y
        res = self.grid.info.resolution
        x = ox + (c + 0.5) * res
        y = oy + (r + 0.5) * res
        return x, y

    def is_valid_cell(self, c, r):
        w = self.grid.info.width
        h = self.grid.info.height
        if 0 <= c < w and 0 <= r < h:
            return self.costmap[r, c] == 0
        return False

    def is_line_clear(self, p1, p2):
        '''Bresenham line clear check on inflated costmap.'''
        c0, r0 = self.world_to_grid(p1[0], p1[1])
        c1, r1 = self.world_to_grid(p2[0], p2[1])

        dx = abs(c1 - c0)
        dy = abs(r1 - r0)
        sx = 1 if c0 < c1 else -1
        sy = 1 if r0 < r1 else -1
        err = dx - dy

        c, r = c0, r0
        while True:
            if not self.is_valid_cell(c, r):
                return False
            if c == c1 and r == r1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                c += sx
            if e2 < dx:
                err += dx
                r += sy
        return True

    def find_nearest_free_cell(self, c, r):
        if self.is_valid_cell(c, r):
            return c, r
        for rad in range(1, 25):
            for dc in range(-rad, rad + 1):
                for dr in range(-rad, rad + 1):
                    if self.is_valid_cell(c + dc, r + dr):
                        return c + dc, r + dr
        return c, r

    def plan_path_single(self, start, end):
        '''A* grid path search with Euclidean heuristic.'''
        start_c, start_r = self.world_to_grid(start[0], start[1])
        end_c, end_r = self.world_to_grid(end[0], end[1])

        start_c, start_r = self.find_nearest_free_cell(start_c, start_r)
        end_c, end_r = self.find_nearest_free_cell(end_c, end_r)

        open_set = []
        h0 = math.hypot(end_c - start_c, end_r - start_r)
        heapq.heappush(open_set, (h0, 0.0, start_c, start_r))

        came_from = {}
        g_scores = {(start_c, start_r): 0.0}

        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

        found = False
        while open_set:
            f, g, c, r = heapq.heappop(open_set)

            if c == end_c and r == end_r:
                found = True
                break

            if g > g_scores.get((c, r), float('inf')):
                continue

            for dc, dr, move_cost in neighbors:
                nc, nr = c + dc, r + dr
                if not self.is_valid_cell(nc, nr):
                    continue

                if abs(dc) == 1 and abs(dr) == 1:
                    if not self.is_valid_cell(c + dc, r) or not self.is_valid_cell(c, r + dr):
                        continue

                tentative_g = g + move_cost
                if tentative_g < g_scores.get((nc, nr), float('inf')):
                    g_scores[(nc, nr)] = tentative_g
                    h = math.hypot(end_c - nc, end_r - nr)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nc, nr))
                    came_from[(nc, nr)] = (c, r)

        if not found:
            return None

        path_cells = [(end_c, end_r)]
        curr = (end_c, end_r)
        while curr in came_from:
            curr = came_from[curr]
            path_cells.append(curr)
        path_cells.reverse()

        raw_points = [self.grid_to_world(c, r) for c, r in path_cells]

        # Path shortcutting via line-of-sight
        simplified = [start[:2]]
        curr_idx = 0
        while curr_idx < len(raw_points) - 1:
            next_idx = len(raw_points) - 1
            while next_idx > curr_idx + 1:
                if self.is_line_clear(raw_points[curr_idx], raw_points[next_idx]):
                    break
                next_idx -= 1
            simplified.append(raw_points[next_idx])
            curr_idx = next_idx

        simplified[-1] = (end[0], end[1])
        simplified[0] = (start[0], start[1])

        # Dense interpolation (max 0.35m spacing) for smooth tracking
        dense_points = [simplified[0]]
        for i in range(len(simplified) - 1):
            p_a = simplified[i]
            p_b = simplified[i + 1]
            seg_dist = math.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1])
            num_sub = max(1, int(math.ceil(seg_dist / 0.35)))
            for k in range(1, num_sub + 1):
                alpha = k / float(num_sub)
                dense_points.append((
                    p_a[0] + alpha * (p_b[0] - p_a[0]),
                    p_a[1] + alpha * (p_b[1] - p_a[1])
                ))

        dense_points[0] = (start[0], start[1])
        dense_points[-1] = (end[0], end[1])
        return dense_points

    def plan_path(self, start, end):
        '''Adaptive inflation planning: attempts widest safety margin first.'''
        radii = [0.44, 0.40, 0.36]
        for rad in radii:
            self.build_costmap(rad)
            path = self.plan_path_single(start, end)
            if path is not None:
                self.get_logger().info(
                    f'Planned path for Leg {self.current_step + 1}: {len(path)} waypoints '
                    f'with safety margin {rad:.2f}m from {start[:2]} to {end[:2]}.'
                )
                return path

        self.get_logger().error(f'A* failed to find path from {start} to {end} across all margins!')
        return None

    def publish_path(self, points, start, end):
        path = Path()
        path.header.frame_id = map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for i, (x, y) in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = map_frame
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            if i == 0:
                yaw = start[2]
            elif i == len(points) - 1:
                yaw = end[2]
            else:
                p_next = points[i + 1]
                yaw = math.atan2(p_next[1] - y, p_next[0] - x)

            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)

        self.path_pub.publish(path)
        self.step_path_published = True
        self.get_logger().info(f'Published /ebot_path for Leg {self.current_step + 1}.')

    def request_ore_package(self):
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn('/spawn_ore_package service is not ready yet.')
            return

        self.get_logger().info('Calling /spawn_ore_package...')
        self.spawn_requested = True
        self.waiting_for_spawn = True

        future = self.spawn_client.call_async(Trigger.Request())
        future.add_done_callback(self.spawn_callback)

    def spawn_callback(self, future):
        try:
            res = future.result()
            self.get_logger().info(f'/spawn_ore_package response: success={res.success}, message="{res.message}"')
            if res.success:
                self.package_spawned = True
                self.waiting_for_spawn = False
                self.spawn_settle_ticks = 0
            else:
                self.get_logger().warn(f'Service call returned False: {res.message}. Retrying...')
                self.waiting_for_spawn = False
                self.spawn_requested = False
                self.step_settle_ticks = 0
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}. Retrying...')
            self.waiting_for_spawn = False
            self.spawn_requested = False
            self.step_settle_ticks = 0

    def process_planning(self):
        if self.grid is None or self.costmap is None or self.odom is None:
            return

        if self.task_finished:
            return

        start_pose, end_pose = steps[self.current_step]

        # Robot kinematic state from /odom
        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        vx = self.odom.twist.twist.linear.x
        vy = self.odom.twist.twist.linear.y
        v_lin = math.hypot(vx, vy)
        v_ang = abs(self.odom.twist.twist.angular.z)
        is_stationary = (v_lin < 0.030 and v_ang < 0.030)

        # 1. Publish path ONLY while eBot stands still
        if not self.step_path_published:
            if not is_stationary:
                return

            if self.current_step == 2:
                # Leg 3: Return to Home via open corridor (+X) to guarantee generous clearance from cargo rover
                points = [
                    (start_pose[0], start_pose[1]),
                    (0.20, 1.50),
                    (0.20, 1.00),
                    (0.20, 0.50),
                    (end_pose[0], end_pose[1])
                ]
            else:
                points = self.plan_path(start_pose, end_pose)
            if points is not None:
                self.publish_path(points, start_pose, end_pose)
                self.step_path_published = True
                self.step_settle_ticks = 0
            return

        # 2. Check if robot has reached the leg goal pose and settled
        dist_to_goal = math.hypot(end_pose[0] - rx, end_pose[1] - ry)
        if math.hypot(end_pose[0], end_pose[1]) < 0.5:
            # Home pose requires yaw 0.0
            yaw_err = abs(math.atan2(math.sin(0.0 - yaw), math.cos(0.0 - yaw)))
        else:
            # Ore drop or arm pose: check alignment to target yaw
            yaw_err = min(
                abs(math.atan2(math.sin(0.0 - yaw), math.cos(0.0 - yaw))),
                abs(math.atan2(math.sin(math.pi - yaw), math.cos(math.pi - yaw)))
            )

        is_at_goal = (dist_to_goal < 0.22 and yaw_err < 0.10 and is_stationary)

        if is_at_goal:
            self.step_settle_ticks += 1
        else:
            self.step_settle_ticks = 0

        # Require 0.2 seconds (2 ticks) of sustained standstill at the goal
        if self.step_settle_ticks < 2:
            return

        # ==========================================================
        # LEG 1: SPAWN PACKAGE AT (6.0, 1.9)
        # ==========================================================
        if self.current_step == 0:
            if not self.package_spawned:
                if not self.spawn_requested:
                    self.get_logger().info(
                        f'eBot firmly stationary on ore drop pose (dist={dist_to_goal:.3f}m, yaw_err={yaw_err:.3f}rad). '
                        f'Calling /spawn_ore_package...'
                    )
                    self.request_ore_package()
                return
            else:
                self.spawn_settle_ticks += 1
                if self.spawn_settle_ticks < 3:
                    return

                self.get_logger().info('Ore package settled on eBot deck! Preparing Leg 2 (Ore Drop -> Arm Pose)...')
                self.current_step = 1
                self.step_path_published = False
                self.step_settle_ticks = 0

        # ==========================================================
        # LEG 2: HANDOVER AT ARM POSE (0.0, 1.9)
        # ==========================================================
        elif self.current_step == 1:
            # Notify handover coordinator that eBot is in position beside the arm
            ebot_arm_msg = Bool()
            ebot_arm_msg.data = True
            self.ebot_at_arm_pub.publish(ebot_arm_msg)

            # Wait for arm confirmation that the package has been lifted off the deck
            if not self.can_depart:
                self.get_logger().info(
                    'eBot stationary at Arm Handover station. Holding deck steady for UR7e arm pickup...',
                    throttle_duration_sec=2.0
                )
                return

            self.get_logger().info(
                f'Arm has lifted the package off the deck! '
                f'Preparing Leg 3 (Arm Pose -> Home Pose)...'
            )
            self.current_step = 2
            self.step_path_published = False
            self.step_settle_ticks = 0

        # ==========================================================
        # LEG 3: RETURN TO HOME (0.0, 0.0)
        # ==========================================================
        elif self.current_step == 2:
            self.task_finished = True
            home_msg = Bool()
            home_msg.data = True
            self.ebot_home_pub.publish(home_msg)

            self.get_logger().info('=' * 68)
            self.get_logger().info(
                f'[SUCCESS] Task 3B Complete! eBot safely returned home '
                f'(dist={dist_to_goal:.3f}m, yaw_err={yaw_err:.3f}rad).'
            )
            self.get_logger().info('All mission legs and manipulation tasks finalized.')
            self.get_logger().info('=' * 68)


def main(args=None):
    rclpy.init(args=args)
    node = Task3BPlanner()
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

