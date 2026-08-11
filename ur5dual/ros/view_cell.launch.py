#!/usr/bin/env python3
"""
Draw the cell in RViz, live.

    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch ur5dual/ros/view_cell.launch.py

Geometry comes from config/cell.yaml, so this shows the cell as configured —
edit the mounting numbers, relaunch, and the model moves with them.

Read-only. Nothing here commands a robot.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, REPO_ROOT)

from ur5dual.config import DEFAULT_PATH                       # noqa: E402
from ur5dual.ros.launch_support import xacro_command_string   # noqa: E402

RVIZ_CONFIG = os.path.join(PKG_DIR, "description", "cell.rviz")


def generate_launch_description():
    config_path = os.environ.get("UR5DUAL_CONFIG", DEFAULT_PATH)
    # resolved at launch time, not at every xacro re-evaluation, so the URDF
    # and the running control code cannot drift apart mid-session
    urdf = ParameterValue(Command([xacro_command_string(config_path=config_path)]),
                          value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("config", default_value=config_path),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": urdf}],
        ),
        Node(
            executable=os.path.join(PKG_DIR, "ros", "state_publisher.py"),
            name="ur5dual_state_publisher",
            output="screen",
            arguments=["--config", LaunchConfiguration("config")],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", RVIZ_CONFIG] if os.path.exists(RVIZ_CONFIG) else [],
        ),
    ])
