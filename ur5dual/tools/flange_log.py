#!/usr/bin/env python3
"""
Pairing a flange separation you measured by hand with the pose the two arms
were standing in when you measured it.

    scripts/ur5dual-flange-log

Drive the arms wherever you want them with the control panel, put the gauge
across the two axis-6 flanges, type the reading in millimetres, press Enter.
One line goes into config/flange_log.json: that number, and each arm's six
joint angles in degrees and its flange x y z in millimetres. Nothing else,
and nothing moves — this program has no way to command a robot.

The two frames worth being clear about, because mixing them up is a quiet
error of exactly one tool length:

  flange      what UR calls tool0, the face at the end of joint 6. This is
              what gets recorded, because it is what a gauge can be laid
              across, and it is what `ur_kinematics.fk` returns.
  TCP         the flange times whatever tool offset is set on the pendant.
              What the panel and the pendant display. Not recorded here.

Each arm's xyz is in *its own base frame* — the frame its controller speaks,
which nothing in config/cell.yaml can make wrong. That is deliberate: the
usual reason to collect these samples is that the world-frame geometry is
what you are trying to check, and a record expressed in it would move
whenever the file it is being checked against was edited.

Where the joint angles come from:

  the control panel   the default, and nothing is asked of the robots at all.
                      ur5dual-gui already owns port 30003 on both of them and
                      forwards every sample it reads over loopback — 30099 for
                      the RViz bridge to draw, 30100 for readers like this one,
                      which is why both can watch at once. PolyScope 3.7
                      on arm A does not reliably serve two clients on 30003 —
                      the second stalls mid-packet and never resumes, and both
                      readers then quietly keep their last sample — so this is
                      not a preference but the only safe way to watch a cell
                      that is being driven.
  --direct            straight from the two controllers, for a cell with no
                      panel running. Refused while one is publishing, and
                      dropped again the moment one appears.

Joint angles are all the panel's channel carries, so the flange pose is worked
out here from the published UR5 DH table, which describes these two arms to
about 3 mm. --direct also reads the table each controller holds for itself,
which is exact. The file records which was used and refuses to mix the two,
because the difference between them is the size of what is being measured.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np

from ..config import ARM_IDS, DEFAULT_PATH, REPO_ROOT, CellConfig
from ..geometry import ur_kinematics as UK
from ..geometry.kinematics import pose_to_mat
from ..robot.transport import StateStream, read_geometry
from ..sim_view import (
    SIM_VIEW_HOST, SIM_VIEW_PORT, SIM_VIEW_TAP_PORT, SimViewReceiver,
)

DEFAULT_LOG = os.path.join(REPO_ROOT, "config", "flange_log.json")

PUBLISHED_DH = "the published UR5 table"

# What the distance in a sample means. A file holds one of these and never a
# mixture: the same six angles and one number describe a different measurement
# under each, a fit reads every sample in a file the same way, and nothing in
# the numbers themselves would ever show the difference.
SEPARATION = "separation"   # centre to centre, which is what a gauge reads
FACING = "facing"           # between parallel faces, which is what a block of
                            # known thickness holds them at

# A gauge reading and a pose that was taken while the arm was still moving
# describe two different moments. At a jog speed of 50 mm/s, the 300 ms it
# takes to notice is 15 mm of pure fiction — larger than anything this file is
# collected to measure. So both arms have to be standing still, and 0.05 deg
# over that interval is well inside encoder noise.
SETTLE = 0.30           # s between the two readings that decide it
STILL_DEG = 0.05        # deg of joint motion allowed across them

# How long to wait at start-up for a control panel to announce itself. Its
# heartbeat is one datagram every 60 ms, so this is generous; it only has to
# survive the GUI pausing to repaint.
PANEL_WAIT = 1.0


# ── where the joint angles come from ──────────────────────────────────────
def channel_is_free(host=SIM_VIEW_HOST, port=SIM_VIEW_PORT):
    """True when nothing else is listening for the panel's datagrams.

    Worth a probe rather than simply trying, because the obvious attempt gives
    the wrong answer. Linux lets a second socket bind a UDP port that is
    already held, provided both set SO_REUSEADDR — and then delivers every
    datagram to the newcomer, leaving the original with none. RViz's receiver
    sets that option, so binding on top of it does not share the channel, it
    takes it: RViz sees silence, concludes from it that no control panel is
    running, and opens its own connection to each robot. That is precisely the
    second client on port 30003 this program exists not to be, arrived at by
    a program that never touched a robot.

    A plain bind, with the option left off, is refused while anyone holds the
    port. So it answers the question and, being closed immediately, never
    becomes the problem it is asking about.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


class PanelFeed:
    """The angles the control panel has already read from the two robots.

    Deliberately not a second client of port 30003. The panel forwards every
    sample it reads over loopback — 30100 for readers like this one, 30099 for
    the RViz bridge — so this sees exactly what the panel sees and costs the
    controllers nothing.

    Both ports are listened to, each only while nothing else holds it. The
    viewer's is worth trying because a panel that has not been restarted since
    the tap was added still publishes on 30099 alone, and on a bench with no
    RViz running there is no reason that should stop the recording.

    Draining runs in its own thread, and has to. This program spends nearly
    all of its life blocked in `input()` waiting for somebody to read a gauge,
    and a socket nobody is reading fills up in about a minute at 16 datagrams
    a second. A full UDP receive buffer does not drop the oldest to make room
    for the newest — it drops the newest and keeps the queue it has, so the
    freshest frame available becomes the one from the moment it filled, and
    every later one is thrown away by the kernel. Reading the queue then
    reports a panel that stopped publishing a minute ago while the panel is
    right there publishing, which is precisely what it did.
    """

    source = "the control panel's feed"
    POLL = 0.02             # s between drains; the panel sends every 8 to 60 ms

    def __init__(self, host=SIM_VIEW_HOST,
                 ports=(SIM_VIEW_TAP_PORT, SIM_VIEW_PORT)):
        self.receivers = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        taken = []
        for port in ports:
            if channel_is_free(host, port):
                self.receivers.append(SimViewReceiver(host, port))
            else:
                taken.append(port)
        if not self.receivers:
            self.close()
            raise OSError("%s already listening"
                          % " and ".join("%s:%d" % (host, p) for p in taken))
        self.ports = [rx.sock.getsockname()[1] for rx in self.receivers]
        self.dh = {a: UK.UR5_DH for a in ARM_IDS}
        self.dh_source = {a: PUBLISHED_DH for a in ARM_IDS}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.POLL):
            self._drain()

    def _drain(self):
        """Every port drained, in the order they are preferred."""
        with self._lock:
            return [rx.poll() for rx in self.receivers]

    @property
    def live(self):
        """True while a panel is publishing — connected arms or not."""
        self._drain()
        with self._lock:
            return any(rx.active for rx in self.receivers)

    def joints(self):
        got = self._drain()
        with self._lock:
            for rx, joints in zip(self.receivers, got):
                if joints and rx.mode != "sim":
                    return {a: np.array(q, dtype=float)
                            for a, q in joints.items() if a in ARM_IDS}
        return {}

    def complaint(self, missing):
        if not any(rx.active for rx in self.receivers):
            return ("the control panel has stopped publishing — it has been "
                    "closed, or it predates the reader's port (%d) and the "
                    "RViz bridge has the other one; restarting the panel gives "
                    "this program its own"
                    % SIM_VIEW_TAP_PORT)
        if any(rx.mode == "sim" for rx in self.receivers):
            return ("the control panel is in SIM — those joint angles are "
                    "computed, not measured, and pairing them with a gauge "
                    "reading would record a robot that was never there")
        return ("the control panel has no connection to arm %s — press "
                "Connect for it there" % " or ".join(missing))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        for rx in self.receivers:
            rx.close()
        self.receivers = []


class DirectFeed:
    """Straight from the two controllers, for when no panel is running.

    Worth having beyond mere convenience: each controller also hands over the
    DH table measured for that particular arm at the factory, so the flange
    positions recorded through here are the machine's own answer rather than
    the drawing's.
    """

    source = "the robots' own feeds"

    def __init__(self, config, arm_ids=ARM_IDS, timeout=5.0):
        self.streams = {}
        self.dh = {}
        self.dh_source = {}
        try:
            for arm_id in arm_ids:
                ip = config.arms[arm_id].ip
                stream = StateStream(ip, timeout=timeout)
                self.streams[arm_id] = stream
                _, offset, calibration = read_geometry(ip, timeout=timeout)
                state = stream.latest()
                dh, source, _ = UK.choose_dh(
                    np.array(state["q_actual"], dtype=float), state["tcp_pose"],
                    None if offset is None else pose_to_mat(offset), calibration)
                self.dh[arm_id] = dh
                self.dh_source[arm_id] = source
        except (OSError, ConnectionError):
            self.close()
            raise

    def joints(self):
        out = {}
        for arm_id, stream in self.streams.items():
            try:
                out[arm_id] = np.array(stream.latest()["q_actual"], dtype=float)
            except (OSError, ConnectionError, RuntimeError):
                continue
        return out

    def complaint(self, missing):
        return "arm %s stopped answering" % " and ".join(missing)

    def close(self):
        for stream in self.streams.values():
            stream.close()
        self.streams = {}


class Feeds:
    """Whichever of the two is right at this moment, and the handover.

    The handover is the point. An operator who starts this program first and
    the panel second must not end up with two clients on port 30003, and
    telling them so afterwards is no use — by then the panel's feed is the one
    that has stalled. So the panel is watched the whole time, and these
    connections are let go the instant one appears.

    Which is also why not being able to listen at all — both loopback ports
    held by other programs — is treated as a reason to touch nothing. With the
    one channel that could have said whether a panel is driving this cell
    unavailable, "probably nobody is connected" is not a good enough basis for
    opening a second feed to a robot. Saying so and stopping is.
    """

    def __init__(self, config, direct=False, log=print):
        self.config = config
        self.wants_direct = direct
        self.log = log
        self.direct = None
        self.panel = None
        self.panel_error = None
        try:
            self.panel = PanelFeed()
        except OSError as e:
            self.panel_error = (
                "cannot listen for a control panel — %s. Another copy of this "
                "program has the reader's port (%d) and the RViz bridge has "
                "the viewer's (%d); close one of them, or pass --direct if you "
                "are certain no panel is running"
                % (e, SIM_VIEW_TAP_PORT, SIM_VIEW_PORT))

    def current(self):
        """(feed, why there is none). Never opens a robot feed under a panel."""
        if self.panel is not None and self.panel.live:
            if self.direct is not None:
                self.direct.close()
                self.direct = None
                self.log("a control panel has started — handing the robots "
                         "back to it and reading its feed instead")
            if self.wants_direct:
                return None, ("a control panel is publishing, and it owns port "
                              "30003 on both robots — close it, or drop "
                              "--direct and this will read what it reads")
            return self.panel, None

        if not self.wants_direct:
            if self.panel is None:
                return None, self.panel_error
            return None, ("nothing is publishing on the panel's channel — "
                          "start scripts/ur5dual-gui and connect both arms "
                          "there, or read the two robots directly with "
                          "--direct")

        if self.direct is None:
            try:
                self.direct = DirectFeed(self.config)
            except (OSError, ConnectionError) as e:
                return None, "the robots will not answer: %s" % e
            self.log("reading the two robots directly (%s / %s)"
                     % tuple(self.config.arms[a].ip for a in ARM_IDS))
        return self.direct, None

    def close(self):
        for feed in (self.panel, self.direct):
            if feed is not None:
                feed.close()


# ── the measurement ───────────────────────────────────────────────────────
def flange_matrix(q, dh):
    """Joint angles -> 4x4 flange pose in that arm's own base frame."""
    return UK.fk(np.asarray(q, dtype=float), dh)


def steady_joints(feed, settle=SETTLE, still_deg=STILL_DEG):
    """Both arms' angles, once they have stopped. (joints, why not)."""
    first = feed.joints()
    missing = [a for a in ARM_IDS if a not in first]
    if missing:
        return None, feed.complaint(missing)

    time.sleep(settle)
    second = feed.joints()
    missing = [a for a in ARM_IDS if a not in second]
    if missing:
        return None, feed.complaint(missing)

    for arm_id in ARM_IDS:
        moved = float(np.max(np.abs(np.degrees(second[arm_id] - first[arm_id]))))
        if moved > still_deg:
            return None, ("arm %s moved %.2f deg while this was being read — "
                          "let go of the jog and try again" % (arm_id, moved))
    return second, None


def build_sample(gap_mm, joints, dh):
    """One line of the log: the gauge, and where each arm was."""
    sample = {"t": datetime.now().isoformat(timespec="seconds"),
              "gap_mm": round(float(gap_mm), 2)}
    for arm_id in ARM_IDS:
        xyz = flange_matrix(joints[arm_id], dh[arm_id])[:3, 3] * 1000.0
        sample[arm_id] = {
            "joints_deg": [round(float(v), 4)
                           for v in np.degrees(joints[arm_id])],
            "xyz_mm": [round(float(v), 3) for v in xyz],
        }
    return sample


def configured_gap_mm(config, joints, dh):
    """What cell.yaml says the flanges are apart, for the same joint angles.

    Shown next to the gauge, never written to the file. It is the number the
    measurement exists to check, and it changes whenever anybody edits the
    mounting geometry — a record that moved with it would be worthless.
    """
    if any(a not in joints for a in ARM_IDS):
        return None
    world = {a: (config.arms[a].base_matrix()
                 @ flange_matrix(joints[a], dh[a]))[:3, 3] for a in ARM_IDS}
    return float(np.linalg.norm(world["A"] - world["B"]) * 1000.0)


# ── the file ──────────────────────────────────────────────────────────────
def dh_numbers(dh):
    """A DH table as plain lists, so a later fit can use the one that was read.

    The published table and a controller's own differ by about 3 mm on these
    two arms, which is the size of what these samples are collected to find.
    The file already names which was used; carrying the numbers as well is
    what lets an offline fit reproduce those poses rather than approximate
    them from the drawing.
    """
    return {k: [float(v) for v in np.asarray(dh[k], dtype=float)]
            for k in UK.DH_FIELDS}


def log_meta(source, dh_source, dh, model=SEPARATION):
    """Everything about a session that the samples themselves do not say."""
    return {
        "units": "mm and degrees",
        "frame": "flange (tool0), in each arm's own base frame",
        "source": source,
        "model": model,
        "kinematics": {a: dh_source[a] for a in ARM_IDS},
        "dh": {a: dh_numbers(dh[a]) for a in ARM_IDS},
    }


def meta_for(feed):
    return log_meta(feed.source, feed.dh_source, feed.dh)


def load_log(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"meta": {}, "samples": []}
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise SystemExit("%s is not a flange log — point --file somewhere else"
                         % path)
    data.setdefault("meta", {})
    return data


def save_log(path, data):
    """Written whole, every time, and renamed into place.

    A session is a slow accumulation of readings somebody took by hand, so
    each one is on disk before the next is asked for; and the rename means an
    interrupted write cannot lose the ones already taken.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def kinematics_clash(data, meta):
    """Why these samples must not go in this file, or None.

    The published table and a controller's own differ by about 3 mm on these
    two arms. That is the same size as the errors this file is collected to
    find, so a file holding some of each cannot answer the question it was
    written for — and nothing in the numbers themselves would show it.
    """
    was = data["meta"].get("kinematics")
    if not was or was == meta["kinematics"]:
        return None
    return ("what is in this file was computed from %s; this session would "
            "add samples computed from %s. The two differ by millimetres, "
            "which is the size of what you are measuring — write this "
            "session to another file"
            % (" / ".join(sorted(set(was.values()))),
               " / ".join(sorted(set(meta["kinematics"].values())))))


def model_clash(data, meta):
    """Why these samples mean something else than the ones already here.

    A gauge laid across the two flange centres and a block held between the
    two flange faces are each one distance and a pair of postures, and they
    are not the same measurement. A fit applies whichever model it is given to
    every sample in the file, so a file that holds both is wrong about half of
    them whichever way it is read — and the numbers look identical either way.
    """
    if not data["samples"]:
        return None
    was = data["meta"].get("model") or SEPARATION
    if was == meta["model"]:
        return None
    return ("this file holds %d %s sample%s and this session records %s ones. "
            "A fit reads every sample in a file the same way, so the two "
            "cannot share one — write this session somewhere else"
            % (len(data["samples"]), was,
               "" if len(data["samples"]) == 1 else "s", meta["model"]))


# ── the terminal ──────────────────────────────────────────────────────────
def show(config, feed, joints):
    lines = []
    for arm_id in ARM_IDS:
        xyz = flange_matrix(joints[arm_id], feed.dh[arm_id])[:3, 3] * 1000.0
        lines.append("%s  J %s deg"
                     % (arm_id, " ".join("%8.2f" % v
                                         for v in np.degrees(joints[arm_id]))))
        lines.append("   flange  x %9.2f  y %9.2f  z %9.2f mm   in arm %s's "
                     "base" % (xyz[0], xyz[1], xyz[2], arm_id))
    gap = configured_gap_mm(config, joints, feed.dh)
    if gap is not None:
        lines.append("   cell.yaml puts the two flanges %.1f mm apart" % gap)
    print("\n".join(lines))


def describe(sample):
    return ("%6.1f mm gap   A %s   B %s"
            % (sample["gap_mm"],
               " ".join("%.1f" % v for v in sample["A"]["joints_deg"]),
               " ".join("%.1f" % v for v in sample["B"]["joints_deg"])))


HELP = """
  <number>   the gauge reading in mm — records both arms as they stand now
  Enter      show where the arms are without recording
  l          list what has been recorded
  u          undo the last one
  q          quit
"""


class Recorder:
    def __init__(self, config, path, feeds):
        self.config = config
        self.path = path
        self.feeds = feeds
        self.data = load_log(path)

    def run(self):
        print("recording to %s — %d sample%s already there"
              % (self.path, len(self.data["samples"]),
                 "" if len(self.data["samples"]) == 1 else "s"))
        print(HELP)
        while True:
            try:
                answer = input("flange gap (mm) > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if answer in ("q", "quit", "exit"):
                return
            if answer in ("l", "list"):
                self.list_samples()
            elif answer in ("u", "undo"):
                self.undo()
            elif answer in ("?", "h", "help"):
                print(HELP)
            elif answer == "":
                self.show_now()
            else:
                self.record(answer)

    # -- what the prompt does ---------------------------------------------
    def feed_and_joints(self):
        feed, why = self.feeds.current()
        if feed is None:
            print(why)
            return None, None
        joints, why = steady_joints(feed)
        if joints is None:
            print(why)
            return None, None
        return feed, joints

    def show_now(self):
        feed, joints = self.feed_and_joints()
        if joints is not None:
            show(self.config, feed, joints)

    def record(self, answer):
        try:
            gap_mm = float(answer)
        except ValueError:
            print("that is not a number of millimetres, and not a command "
                  "either — press ? for the list")
            return
        if gap_mm <= 0:
            print("a flange separation is a distance in mm, so it is positive")
            return

        feed, joints = self.feed_and_joints()
        if joints is None:
            return

        meta = meta_for(feed)
        for clash in (kinematics_clash(self.data, meta),
                      model_clash(self.data, meta)):
            if clash:
                print(clash)
                return

        sample = build_sample(gap_mm, joints, feed.dh)
        self.data["meta"] = meta
        self.data["samples"].append(sample)
        save_log(self.path, self.data)
        show(self.config, feed, joints)
        print("recorded %d: %.1f mm apart, both arms, from %s"
              % (len(self.data["samples"]), sample["gap_mm"], feed.source))

    def list_samples(self):
        if not self.data["samples"]:
            print("nothing recorded yet")
            return
        for i, sample in enumerate(self.data["samples"], 1):
            print("%3d  %s" % (i, describe(sample)))

    def undo(self):
        if not self.data["samples"]:
            print("nothing recorded yet")
            return
        dropped = self.data["samples"].pop()
        save_log(self.path, self.data)
        print("dropped %s" % describe(dropped))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_LOG,
                    help="where to keep the samples (default %s)" % DEFAULT_LOG)
    ap.add_argument("--config", default=DEFAULT_PATH)
    ap.add_argument("--direct", action="store_true",
                    help="read the two robots' own feeds instead of a control "
                         "panel's — only for a cell with no panel running")
    args = ap.parse_args()

    if not sys.stdin.isatty():
        raise SystemExit("this program is a prompt — run it in a terminal.")

    config = CellConfig.load(args.config)
    feeds = Feeds(config, direct=args.direct)
    if feeds.panel is None:
        print(feeds.panel_error)
    else:
        print("listening for the control panel on %s"
              % ", ".join("%s:%d" % (SIM_VIEW_HOST, p) for p in feeds.panel.ports))
        # a panel that is already running is one heartbeat away at most, and
        # deciding it is absent before that arrives would be wrong in the one
        # direction that matters
        deadline = time.time() + PANEL_WAIT
        while not feeds.panel.live and time.time() < deadline:
            time.sleep(0.05)
    try:
        Recorder(config, args.file, feeds).run()
    finally:
        feeds.close()


if __name__ == "__main__":
    main()
