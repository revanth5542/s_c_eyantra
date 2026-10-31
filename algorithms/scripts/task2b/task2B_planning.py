#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  Task 2B: Path Planning, Obstacle Avoidance and Ore Package Delivery.
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
from std_srvs.srv import Trigger


# Topics and Frames
map_topic = '/map'
path_topic = '/ebot_path'
odom_topic = '/odom'
spawn_service = '/spawn_ore_package'
map_frame = 'map'

# Poses of the lap: (x, y, yaw)
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)
arm_pose = (0.0, 1.9, math.pi)

steps = [
    (home_pose, ore_drop_pose),     # step 0: -> request ore package
    (ore_drop_pose, arm_pose),      # step 1: with package -> arm
    (arm_pose, home_pose),          # step 2: -> back home
]

# Obstacle inflation radius (metres) for eBot footprint safety
INFLATION_RADIUS = 0.36


class EbotPlanner(Node):
    def __init__(self):
        super().__init__(
            'ebot_planner_node',
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

        self.grid = None
        self.costmap = None
        self.odom = None
        self.current_step = 0
        self.step_path_published = False
        self.waiting_for_spawn = False
        self.spawn_requested = False
        self.package_spawned = False
        self.step_settle_ticks = 0
        self.spawn_settle_ticks = 0
        self.task_finished = False

        self.timer = self.create_timer(0.1, self.process_planning)
        self.get_logger().info('Task 2B Planner Node Initialized.')

    def mapcb(self, data: OccupancyGrid):
        if self.grid is None:
            self.grid = data
            self.build_costmap()
            self.get_logger().info(
                f'Map received: {data.info.width}x{data.info.height} cells @ '
                f'{data.info.resolution:.3f} m/cell, origin=({data.info.origin.position.x:.2f}, '
                f'{data.info.origin.position.y:.2f})'
            )

    def build_costmap(self):
        w = self.grid.info.width
        h = self.grid.info.height
        res = self.grid.info.resolution

        raw_data = np.array(self.grid.data, dtype=np.int8).reshape((h, w))
        # Occupied cells: occupied if > 50
        occupied_mask = (raw_data > 50)

        # Distance transform to nearest obstacle (in metres)
        dist_cells = distance_transform_edt(~occupied_mask)
        dist_m = dist_cells * res

        # Invalidate any cell closer than INFLATION_RADIUS
        self.costmap = np.zeros((h, w), dtype=np.uint8)
        self.costmap[dist_m < INFLATION_RADIUS] = 100
        self.get_logger().info(f'Costmap built with inflation radius {INFLATION_RADIUS:.2f} m.')

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
        '''Finds nearest traversable cell if start/goal is within inflation zone.'''
        if self.is_valid_cell(c, r):
            return c, r
        for rad in range(1, 20):
            for dc in range(-rad, rad + 1):
                for dr in range(-rad, rad + 1):
                    if self.is_valid_cell(c + dc, r + dr):
                        return c + dc, r + dr
        return c, r

    def plan_path(self, start, end):
        '''A* grid path search with Euclidean heuristic.'''
        start_c, start_r = self.world_to_grid(start[0], start[1])
        end_c, end_r = self.world_to_grid(end[0], end[1])

        start_c, start_r = self.find_nearest_free_cell(start_c, start_r)
        end_c, end_r = self.find_nearest_free_cell(end_c, end_r)

        # Priority queue: (f_score, g_score, c, r)
        open_set = []
        h0 = math.hypot(end_c - start_c, end_r - start_r)
        heapq.heappush(open_set, (h0, 0.0, start_c, start_r))

        came_from = {}
        g_scores = {(start_c, start_r): 0.0}

        # 8-connected neighbors: (dc, dr, cost)
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

                # Corner cutting check for diagonal moves
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
            self.get_logger().error(f'A* failed to find path from {start} to {end}!')
            return None

        # Reconstruct grid cell path
        path_cells = [(end_c, end_r)]
        curr = (end_c, end_r)
        while curr in came_from:
            curr = came_from[curr]
            path_cells.append(curr)
        path_cells.reverse()

        # Convert to world coordinates
        raw_points = [self.grid_to_world(c, r) for c, r in path_cells]

        # Path simplification: shortcut clear line of sight
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

        # Enforce exact end pose coordinates
        simplified[-1] = (end[0], end[1])
        # Enforce exact start pose coordinates
        simplified[0] = (start[0], start[1])

        # Interpolate points so spacing is at most 0.40m for smooth follower tracking
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

        self.get_logger().info(
            f'Planned path for step {self.current_step + 1}: {len(dense_points)} waypoints '
            f'from {start[:2]} to {end[:2]}.'
        )
        return dense_points

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
        self.get_logger().info(f'Published /ebot_path for step {self.current_step + 1}.')

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
        is_stationary = (v_lin < 0.015 and v_ang < 0.015)

        # 1. If path for current step not yet published, plan and publish ONLY while eBot stands still
        if not self.step_path_published:
            if not is_stationary:
                # Wait for eBot to come to a complete standstill before publishing step path
                return

            points = self.plan_path(start_pose, end_pose)
            if points is not None:
                self.publish_path(points, start_pose, end_pose)
                self.step_path_published = True
                self.step_settle_ticks = 0
            return

        # 2. Check if robot has reached the step goal pose and settled
        dist_to_goal = math.hypot(end_pose[0] - rx, end_pose[1] - ry)
        if math.hypot(end_pose[0], end_pose[1]) < 0.5:
            # Home pose requires yaw 0.0
            yaw_err = abs(math.atan2(math.sin(0.0 - yaw), math.cos(0.0 - yaw)))
        else:
            # Ore drop or arm pose: both 0.0 and pi are valid
            yaw_err = min(
                abs(math.atan2(math.sin(0.0 - yaw), math.cos(0.0 - yaw))),
                abs(math.atan2(math.sin(math.pi - yaw), math.cos(math.pi - yaw)))
            )

        # Tolerance check: well within evaluator's 0.30m and 0.15 rad limits
        is_at_goal = (dist_to_goal < 0.20 and yaw_err < 0.08 and is_stationary)

        if is_at_goal:
            self.step_settle_ticks += 1
        else:
            self.step_settle_ticks = 0

        # Require at least 1.0 second (10 ticks) of sustained standstill at the goal
        if self.step_settle_ticks < 10:
            return

        # Handle step-specific completion actions
        if self.current_step == 0:
            # Step 1 Destination reached (Ore Drop Pose)
            if not self.package_spawned:
                if not self.spawn_requested:
                    self.get_logger().info(
                        f'eBot firmly stationary on ore drop pose (dist={dist_to_goal:.3f}m, yaw_err={yaw_err:.3f}rad). '
                        f'Calling /spawn_ore_package...'
                    )
                    self.request_ore_package()
                return
            else:
                # Package granted! Wait 3.0 seconds (30 ticks) for package to drop and settle on the eBot deck
                self.spawn_settle_ticks += 1
                if self.spawn_settle_ticks < 30:
                    return

                self.get_logger().info('Ore package settled on eBot deck! Preparing Step 2 path (Ore Drop -> Arm Pose)...')
                self.current_step = 1
                self.step_path_published = False
                self.step_settle_ticks = 0

        elif self.current_step == 1:
            # Step 2 Destination reached (Arm Pose)
            self.get_logger().info(
                f'Step 2 complete! Arrived at Arm Pose (dist={dist_to_goal:.3f}m, yaw_err={yaw_err:.3f}rad). '
                f'Preparing Step 3 path (Arm -> Home)...'
            )
            self.current_step = 2
            self.step_path_published = False
            self.step_settle_ticks = 0

        elif self.current_step == 2:
            # Step 3 Destination reached (Home Pose)
            self.task_finished = True
            self.get_logger().info('=' * 68)
            self.get_logger().info(
                f'[SUCCESS] Task 2B Completed! eBot safely returned home '
                f'(dist={dist_to_goal:.3f}m, yaw_err={yaw_err:.3f}rad).'
            )
            self.get_logger().info('You may now press "q" in the evaluator terminal to record the score!')
            self.get_logger().info('=' * 68)


def main(args=None):
    rclpy.init(args=args)
    node = EbotPlanner()
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
