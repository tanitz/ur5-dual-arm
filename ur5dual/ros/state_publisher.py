#!/usr/bin/env python3
"""
Both arms' joint angles as /joint_states, so RViz draws the cell.

Three sources, in order of precedence:

  panel        loopback datagrams from the control panel in SIM or REAL.  In
               REAL this reuses the panel's readings instead of opening a
               second connection to an older controller.
  the robots   each arm's own 125 Hz feed, used only when no panel is running.
  zero         every joint at 0, for an arm that is disabled, unreachable, or
               simply not switched on yet.

Never blocks waiting for a robot. The viewer comes up immediately with the
cell drawn straight — arms out along their own x, which is the zero pose and
is obviously a placeholder rather than a plausible reading — and each arm
snaps to its real angles when its feed opens. An arm that is missing is still
published, because a joint absent from the message makes robot_state_publisher
drop the whole branch and the arm vanishes from the scene entirely, which
looks like a broken model rather than a robot that is off.

Read-only with respect to the robots.  Its direct feeds are fallback-only and
are closed as soon as a control panel announces itself.
"""

import argparse
import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ur5dual.config import ARM_IDS, DEFAULT_PATH, CellConfig
from ur5dual.sim_view import SimViewReceiver
from ur5dual.robot.transport import StateStream

UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# Every joint at zero. Not a pose any arm would be parked in, which is the
# point: it reads as "no reading" at a glance, where a plausible-looking
# folded pose reads as a robot that is simply standing there.
ZERO_POSE = (0.0,) * 6

# How long to wait for a robot that may not be there. The viewer's job is to
# draw the cell, and it can do that with no robots at all — so a connect
# attempt that would stall the whole node for five seconds per arm, twice a
# retry cycle, is the wrong trade. Anything reachable on this network answers
# far inside this.
CONNECT_TIMEOUT = 0.4


class CellStatePublisher(Node):
    def __init__(self, config_path, rate_hz):
        super().__init__("ur5dual_state_publisher")
        self.cfg = CellConfig.load(config_path)
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.streams = {}
        self.warned = set()
        self.sim = SimViewReceiver()
        self.panel_active = False

        for a in ARM_IDS:
            if not self.cfg.arms[a].enabled:
                self.get_logger().info(
                    "arm %s disabled — drawn at the zero pose" % a)

        self.get_logger().info(
            "listening for the control panel on udp/%d; SIM and REAL joint "
            "states use this channel so RViz does not duplicate robot feeds"
            % self.sim.sock.getsockname()[1])

        self.create_timer(1.0 / rate_hz, self._tick)
        # Give an already-running panel time to announce itself before opening
        # fallback port-30003 connections during RViz startup.
        self.create_timer(1.0, self._retry)

    def _open(self, arm_id):
        try:
            self.streams[arm_id] = StateStream(self.cfg.arms[arm_id].ip,
                                               timeout=CONNECT_TIMEOUT)
            self.get_logger().info("arm %s reading %s:30003" % (
                arm_id, self.cfg.arms[arm_id].ip))
            self.warned.discard(arm_id)
            return True
        except OSError as e:
            if arm_id not in self.warned:
                self.get_logger().warn(
                    "arm %s not answering (%s) — drawn at the zero pose, will "
                    "keep trying" % (arm_id, e))
                self.warned.add(arm_id)
            return False

    def _retry(self):
        if self.sim.active:
            return
        for a in ARM_IDS:
            if self.cfg.arms[a].enabled and a not in self.streams:
                self._open(a)

    def _tick(self):
        panel_joints = self.sim.poll()
        active = self.sim.active
        if active and self.streams:
            for stream in self.streams.values():
                stream.close()
            self.streams.clear()
            self.get_logger().info(
                "control panel active — released RViz fallback robot feeds")
        if active != self.panel_active:
            self.panel_active = active
            if active:
                self.get_logger().info(
                    "drawing %s joint states from the control panel"
                    % self.sim.mode.upper())
            else:
                self.get_logger().info(
                    "control panel silent — falling back to robot feeds")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names, positions, velocities = [], [], []

        for a in ARM_IDS:
            q, qd = ZERO_POSE, (0.0,) * 6
            if active and panel_joints and a in panel_joints:
                # The panel channel carries no velocities and does not need to:
                # RViz draws position, and a made-up velocity would be a number
                # somebody could later mistake for a measurement.
                q = panel_joints[a]
            elif not active:
                stream = self.streams.get(a)
                if stream is not None:
                    try:
                        st = stream.latest()
                        q, qd = st["q_actual"], st["qd_actual"]
                    except (OSError, ConnectionError) as e:
                        self.get_logger().warn("arm %s stream lost: %s" % (a, e))
                        stream.close()
                        self.streams.pop(a, None)
            names += ["%s_%s" % (a, j) for j in UR_JOINTS]
            positions += list(q)
            velocities += list(qd)

        msg.name = names
        msg.position = positions
        msg.velocity = velocities
        self.pub.publish(msg)

    def destroy_node(self):
        for s in self.streams.values():
            s.close()
        self.sim.close()
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_PATH)
    ap.add_argument("--rate", type=float, default=50.0)
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = CellStatePublisher(args.config, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
