#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  Task 3B: Launch File for all Team Solution Nodes.
*  Starts perception, manipulation, planning, navigation, and handover coordination.
*
*  Usage:
*      ros2 launch algorithms task3B_nodes.launch.py
*
*****************************************************************************************
'''

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    sim_time_param = {"use_sim_time": True}

    handover_node = Node(
        package="algorithms",
        executable="task3B_handover.py",
        name="task3B_handover",
        parameters=[sim_time_param],
        output="screen",
    )

    perception_node = Node(
        package="algorithms",
        executable="task3B_perception.py",
        name="task3B_perception",
        parameters=[sim_time_param],
        output="screen",
    )

    manipulation_node = Node(
        package="algorithms",
        executable="task3B_manipulation.py",
        name="task3B_manipulation",
        parameters=[sim_time_param],
        output="screen",
    )

    planning_node = Node(
        package="algorithms",
        executable="task3B_planning.py",
        name="task3B_planning",
        parameters=[sim_time_param],
        output="screen",
    )

    navigation_node = Node(
        package="algorithms",
        executable="task3B_navigation.py",
        name="task3B_navigation",
        parameters=[sim_time_param],
        output="screen",
    )

    return LaunchDescription([
        handover_node,
        perception_node,
        manipulation_node,
        planning_node,
        navigation_node,
    ])

