#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configurer.py
Standalone Inter-Reconfiguration Framework for Scalable Inter-Reconfigurable Robots.

Implements the three-state FSM (Configured / Fusion / Fission) with 2D twist
transformation for cmd_vel -> xfm_vel, as described in:

    Wan et al., "Enabling Framework for Constant Complexity Model in
    Autonomous Inter-Reconfigurable Robots", IEEE Transactions on Automation
    Science and Engineering, 2024.  DOI: 10.1109/TASE.2024.3421533.

DESIGN
------
One Configurer instance == one robot's brain.
Each physical robot (in ROS2) or simulated robot (in this demo) owns its own
Configurer.  The class has ZERO knowledge of transport: it does not import
rospy / rclpy / smach / tf.  All inter-robot communication happens through
a single injected callback:

    on_publish_rcfg(target_robot_id, rcfg_list)

In ROS2 Foxy later, this callback wraps a publisher on /wmX/rcfg.
In the bundled headless demo, a SimBus helper routes the callback to the
target Configurer's ingest_rcfg() in-process -- so N robots can be simulated
on a single macOS / Spyder session without spawning processes.

KEY VARIABLES (Table I in paper)
--------------------------------
d        : translation magnitude  ||rT_b[:2, 2]||
phi      : translation angle      atan2(rT_b[1,2], rT_b[0,2])
rT_b     : 3x3 kinematic relationship matrix (2D planar)
cmd_vel  : input command velocity
xfm_vel  : transformed output velocity per robot
rcfg     : [neighbour_id, size, split_command_code]
R_h      : host / own pose      (from localisation, e.g. AMCL)
R_n      : neighbour pose       (during a fusion event)
n        : singletons in this formation (grows on fusion, resets to 1 on fission)
n_alpha  : singletons in host    (= self.n at time of fusion, paper-faithful)
n_beta   : singletons in joining neighbour (= rcfg[1] once handshake completes)

TRANSITIONS
-----------
CONFIG  -> FUSION   when rcfg[0] != 0
CONFIG  -> FISSION  when rcfg[2] == -1
FUSION  -> CONFIG   when counterpart tag arrives (rcfg[1] != 0)
FISSION -> CONFIG   immediately after resetting to singleton

VELOCITY TRANSFORM (proper 2D rigid-body twist, preserved from sample)
----------------------------------------------------------------------
    xfm.vx = cmd.vx + d * sin(phi) * cmd.omega
    xfm.vy = cmd.vy - d * cos(phi) * cmd.omega
    xfm.omega = cmd.omega

Usage
-----
    from Configurer import Configurer, Twist, Pose, SimBus, FSMState

    bus = SimBus()
    c1  = Configurer(robot_id=1, on_publish_rcfg=bus.publish_rcfg)
    c2  = Configurer(robot_id=2, on_publish_rcfg=bus.publish_rcfg)
    bus.register(c1); bus.register(c2)

    # per tick
    c1.ingest_cmd_vel(Twist(linear_x=0.1, angular_z=0.05))
    c1.ingest_pose(R_h=pose1, R_n=pose2)
    xfm_vel, state = c1.step()

ROS2 Foxy integration (future)
------------------------------
Replace SimBus with an rclpy Node that:
    * subscribes /wmX/cmd_vel  -> c.ingest_cmd_vel
    * subscribes /wmX/rcfg     -> c.ingest_rcfg
    * subscribes /wmX/amcl_pose-> c.ingest_pose
    * calls c.step() in a timer callback
    * publishes /wmX/xfm_vel from the returned xfm_vel
    * sets on_publish_rcfg = lambda tid, r: String_pub_to(tid).publish(str(r))

The Configurer class itself requires no modification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# NB: pygame is imported lazily inside _run_pygame_teleop() so that
# `from Configurer import Configurer, Twist, ...` never pulls pygame/SDL in.


# ============================================================================
#  MESSAGE DATACLASSES  (ROS-free stand-ins for geometry_msgs/Twist, Pose)
# ============================================================================

@dataclass
class Twist:
    """2D planar twist.  Matches geometry_msgs/Twist fields used by sample."""
    linear_x:  float = 0.0
    linear_y:  float = 0.0
    angular_z: float = 0.0


@dataclass
class Pose:
    """2D planar pose.  yaw stored in radians (no quaternion needed)."""
    x:   float = 0.0
    y:   float = 0.0
    yaw: float = 0.0


# ============================================================================
#  FSM STATE ENUM
# ============================================================================

class FSMState(Enum):
    CONFIG  = "config"
    FUSION  = "fusion"
    FISSION = "fission"


# ============================================================================
#  CONFIGURER  (one instance per robot)
# ============================================================================

class Configurer:
    """
    Per-robot inter-reconfiguration FSM.

    Parameters
    ----------
    robot_id : int
        Permanent identifier for this robot (e.g. 1 for /wm1).
    on_publish_rcfg : callable (target_id: int, rcfg: list) -> None, optional
        Called when this robot needs to place an rcfg tag on a neighbour's
        inbox.  In ROS2, wires to a publisher; in SimBus, routes in-process.
        Defaults to a no-op (useful for unit tests of the core FSM).
    verbose : bool, optional
        Print FSM transitions and handshake events.

    Attributes
    ----------
    robot_id  : permanent id
    host_id   : identity of the fused singleton this robot participates in.
                Equals robot_id when split (SS); equals min(R1, R2) after fusion.
    n         : singletons in this robot's current formation (>=1).
    rT_b      : 3x3 kinematic relationship matrix.  Identity when SS.
    rcfg      : [neighbour_id, size, split_command_code] inbox.
    fsm_state : current FSMState.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        robot_id: int,
        on_publish_rcfg: Optional[Callable[[int, List[int]], None]] = None,
        verbose: bool = False,
    ):
        if robot_id <= 0:
            raise ValueError("robot_id must be a positive integer.")

        self.robot_id: int = robot_id
        self.host_id:  int = robot_id   # becomes min(R1,R2) after fusion

        # Singleton state
        self.n:         int        = 1
        self.rT_b:      np.ndarray = np.identity(3)
        self.rcfg:      List[int]  = [0, 0, 0]
        self.fsm_state: FSMState   = FSMState.CONFIG

        # Inputs (populated by ingest_* callbacks)
        self.cmd_vel: Twist = Twist()
        self.R_h:     Pose  = Pose()   # this robot's own pose
        self.R_n:     Pose  = Pose()   # neighbour pose (only during fusion)

        # Transport hook
        self._publish: Callable[[int, List[int]], None] = (
            on_publish_rcfg if on_publish_rcfg is not None else (lambda tid, r: None)
        )

        self.verbose: bool = verbose

        # Internal bookkeeping for verbose tracing
        self._last_xfm_vel: Twist = Twist()
        self._tick_count:   int   = 0

    # ------------------------------------------------------------------
    #  Ingest methods  (ROS callback equivalents)
    # ------------------------------------------------------------------

    def ingest_cmd_vel(self, cmd_vel: Twist) -> None:
        """Stash the latest command velocity (nav stack output)."""
        self.cmd_vel = cmd_vel

    def ingest_rcfg(self, rcfg: List[int]) -> None:
        """
        Stash a reconfiguration command.  Mirrors the rcfg_cb of the sample.
        Accepts a 3-element list [neighbour_id, size, split_command_code].
        """
        if not isinstance(rcfg, (list, tuple)) or len(rcfg) < 3:
            # Invalid -> reset to safe default (matches sample behaviour)
            self.rcfg = [0, 0, 0]
            return
        self.rcfg = [int(rcfg[0]), int(rcfg[1]), int(rcfg[2])]
        if self.verbose:
            print(f"[cfg {self.robot_id}] rcfg <- {self.rcfg}")

    def ingest_pose(
        self,
        R_h: Optional[Pose] = None,
        R_n: Optional[Pose] = None,
    ) -> None:
        """Push current own pose and/or neighbour pose (amcl_cb equivalent)."""
        if R_h is not None:
            self.R_h = R_h
        if R_n is not None:
            self.R_n = R_n

    # ------------------------------------------------------------------
    #  FSM tick  (one call == one state execute)
    # ------------------------------------------------------------------

    def step(self) -> Tuple[Twist, FSMState]:
        """
        Execute one FSM tick.

        Returns
        -------
        xfm_vel : Twist
            Transformed velocity this robot should apply this tick.
            Zero during FUSION/FISSION transitions.
        state   : FSMState
            The FSM state AFTER this tick (i.e. what will execute next tick).
        """
        self._tick_count += 1

        if self.fsm_state == FSMState.CONFIG:
            xfm = self._configured_tick()
        elif self.fsm_state == FSMState.FUSION:
            xfm = self._fusion_tick()
        elif self.fsm_state == FSMState.FISSION:
            xfm = self._fission_tick()
        else:
            xfm = Twist()

        self._last_xfm_vel = xfm
        return xfm, self.fsm_state

    # ------------------------------------------------------------------
    #  State handlers
    # ------------------------------------------------------------------

    def _configured_tick(self) -> Twist:
        """
        CONFIGURED state.  Compute xfm_vel from cmd_vel and rT_b, then check
        rcfg for pending fusion / fission transitions.
        """
        # d, phi from current rT_b (2D planar)
        tx, ty = self.rT_b[0, 2], self.rT_b[1, 2]
        d   = float(np.linalg.norm([tx, ty]))
        phi = math.atan2(ty, tx)

        cmd = self.cmd_vel
        xfm = Twist(
            linear_x  = cmd.linear_x + d * math.sin(phi) * cmd.angular_z,
            linear_y  = cmd.linear_y - d * math.cos(phi) * cmd.angular_z,
            angular_z = cmd.angular_z,
        )

        # Transitions (priority matches sample: fusion, fission, else stay).
        #
        # Stale-tag guard: once fused, both members share the same host_id.
        # In the ROS sample this is handled implicitly by namespace switching
        # (the old /wmX/rcfg subscriber is torn down).  Here we emulate it
        # by ignoring any rcfg[0] that points at our current host -- i.e.
        # a fusion request to "join ourselves".
        if self.rcfg[0] != 0 and self.rcfg[0] != self.host_id:
            if self.verbose:
                print(f"[cfg {self.robot_id}] CONFIG -> FUSION  (rcfg={self.rcfg})")
            self.fsm_state = FSMState.FUSION
        elif self.rcfg[2] == -1:
            if self.verbose:
                print(f"[cfg {self.robot_id}] CONFIG -> FISSION (rcfg={self.rcfg})")
            self.fsm_state = FSMState.FISSION
        else:
            # Drop stale self-targeted tags so they don't linger.
            if self.rcfg[0] == self.host_id:
                self.rcfg = [0, 0, 0]

        return xfm

    def _fusion_tick(self) -> Twist:
        """
        FUSION state.  Symmetric two-party handshake:
          1. Publish own tag  [host_id, n, 0]  to the neighbour's rcfg.
          2. If own rcfg[1] != 0, the neighbour has already published back:
             compute new rT_b, update n and host_id, return to CONFIG.
          3. Otherwise remain in FUSION (caller will call step() again).
        """
        R1 = self.host_id        # me
        R2 = self.rcfg[0]        # the named neighbour
        n_alpha = self.n         # paper-faithful (sample hard-coded 1)

        # 1. Publish tag to neighbour's rcfg
        tag = [self.host_id, self.n, 0]
        self._publish(R2, tag)
        if self.verbose:
            print(f"[cfg {self.robot_id}] FUSION publish tag {tag} -> wm{R2}")

        # 2. Check for counterpart reply
        if self.rcfg[1] == 0:
            # Still waiting on the other side's tag
            return Twist()

        # 3. Counterpart has responded -> complete fusion
        n_beta = self.rcfg[1]
        new_n  = n_alpha + n_beta
        new_rT_b = self._compute_rT_b(self.R_h, self.R_n,
                                      n_alpha, n_beta, R1, R2)

        # Fused singleton adopts the lower-id namespace (canonical host)
        new_host_id = min(R1, R2)

        if self.verbose:
            print(f"[cfg {self.robot_id}] FUSION complete: "
                  f"n={n_alpha}+{n_beta}={new_n}  host=wm{new_host_id}  "
                  f"rT_b_tx={new_rT_b[0,2]:.2f} ty={new_rT_b[1,2]:.2f}")

        self.n         = new_n
        self.host_id   = new_host_id
        self.rT_b      = new_rT_b
        self.rcfg      = [0, 0, 0]
        self.fsm_state = FSMState.CONFIG
        return Twist()

    def _fission_tick(self) -> Twist:
        """
        FISSION state.  Reset to singleton configuration (SS):
        rT_b = I, n = 1, host_id = robot_id, clear rcfg, back to CONFIG.
        """
        if self.verbose:
            print(f"[cfg {self.robot_id}] FISSION -> SS "
                  f"(was host=wm{self.host_id} n={self.n})")

        self.n         = 1
        self.rT_b      = np.identity(3)
        self.host_id   = self.robot_id
        self.rcfg      = [0, 0, 0]
        self.fsm_state = FSMState.CONFIG
        return Twist()

    # ------------------------------------------------------------------
    #  Transform math  (direct port of compute_midpoint_to_robot_transform_2d)
    # ------------------------------------------------------------------

    def _compute_rT_b(
        self,
        R_h: Pose,
        R_n: Pose,
        n_alpha: int,
        n_beta:  int,
        R1: int,
        R2: int,
    ) -> np.ndarray:
        """
        Compute rT_b for this robot given own pose R_h and neighbour pose R_n.

        Asymmetric by host identity:
          * Robot whose id == min(R1, R2) is the canonical host:
            p_m = p_o  -> zero translation -> rT_b = I.
          * The other robot gets translation = vector from host to self,
            expressed in host's frame.

        The weighted midpoint is computed but overwritten (matching the
        sample exactly), because the paper's single-singleton model treats
        the host's frame as the base frame of the formation.
        """
        p_o = np.array([R_h.x, R_h.y], dtype=float)   # own position
        p_t = np.array([R_n.x, R_n.y], dtype=float)   # neighbour position
        theta_o = float(R_h.yaw)                       # own yaw

        # Weighted midpoint (computed-then-overwritten, sample-faithful)
        total_n = float(n_alpha + n_beta)
        _weighted_midpoint = (n_alpha * p_o + n_beta * p_t) / total_n  # noqa

        host = min(R1, R2)
        if self.robot_id == host:
            p_m = p_o          # I am the host -> no offset
        else:
            p_m = p_t          # I am not the host -> offset from host

        v_m = p_o - p_m

        # Rotate into host's frame (sample uses R_o.T @ -v_m)
        c, s = math.cos(theta_o), math.sin(theta_o)
        R_o = np.array([[c, -s], [s, c]])
        v_m_normalized = R_o.T @ (-v_m)

        rT_b = np.identity(3)
        rT_b[:2, 2] = v_m_normalized
        return np.round(rT_b, 2)

    # ------------------------------------------------------------------
    #  Introspection helpers
    # ------------------------------------------------------------------

    def is_split_singleton(self) -> bool:
        """True if this robot is operating independently (SS)."""
        return self.n == 1 and self.host_id == self.robot_id

    def is_fused(self) -> bool:
        """True if part of a fused singleton (FS)."""
        return self.n > 1

    def snapshot(self) -> Dict:
        """Return a dict summary (for logging / viz title / tests)."""
        return {
            "robot_id":  self.robot_id,
            "host_id":   self.host_id,
            "n":         self.n,
            "state":     self.fsm_state.value,
            "rT_b_tx":   float(self.rT_b[0, 2]),
            "rT_b_ty":   float(self.rT_b[1, 2]),
            "d":         float(np.linalg.norm([self.rT_b[0, 2], self.rT_b[1, 2]])),
            "phi":       math.atan2(self.rT_b[1, 2], self.rT_b[0, 2]),
        }


# ============================================================================
#  SIMBUS  (demo-only helper: in-process ROS-topic stand-in)
# ============================================================================

class SimBus:
    """
    In-process message bus for running N Configurer instances on one machine.

    Replaces ROS2 topics for headless / Spyder testing.  Not used in
    production -- ROS2 wrapper will replace this with real publishers.

    Responsibilities
    ----------------
    * Registry of robots (id -> Configurer).
    * Route rcfg publishes to the target Configurer's ingest_rcfg().
    * Demo-time physics: integrate each robot's xfm_vel into a simulated
      ground-truth pose (Euler step), feed back via ingest_pose().
    * Owns the matplotlib visualisation (one fresh figure per frame,
      Spyder-compatible).
    """

    # Colour palette
    COLORS = [
        'blue', 'green', 'red', 'cyan', 'magenta',
        'orange', 'purple', 'brown', 'pink', 'olive',
    ]

    def __init__(self, dt: float = 0.1, visualize: bool = True):
        self.configurers: Dict[int, Configurer] = {}
        self.poses:       Dict[int, Pose]       = {}
        self.dt:          float                 = dt
        self.visualize:   bool                  = visualize
        self._step_ix:    int                   = 0
        self._xfm_log: Dict[int, List[Twist]]   = {}

    # ------------------------------------------------------------------
    #  Registration + transport
    # ------------------------------------------------------------------

    def register(self, configurer: Configurer, pose: Optional[Pose] = None) -> None:
        """Add a Configurer to the bus.  pose = initial ground-truth pose."""
        rid = configurer.robot_id
        self.configurers[rid] = configurer
        self.poses[rid]       = pose if pose is not None else Pose()
        self._xfm_log[rid]    = []

    def publish_rcfg(self, target_id: int, rcfg: List[int]) -> None:
        """Route an rcfg publish from one Configurer to another."""
        target = self.configurers.get(target_id)
        if target is None:
            return  # target not registered -- silently drop (matches flaky topic)
        target.ingest_rcfg(rcfg)

    # ------------------------------------------------------------------
    #  Commands (joystick / teleop equivalents)
    # ------------------------------------------------------------------

    def send_cmd_vel(self, robot_id: int, cmd: Twist) -> None:
        cfg = self.configurers.get(robot_id)
        if cfg is not None:
            cfg.ingest_cmd_vel(cmd)

    def send_fusion_command(self, host_id: int, neighbour_id: int) -> None:
        """Joystick-Y-button equivalent: tell host_id to fuse with neighbour_id."""
        cfg = self.configurers.get(host_id)
        if cfg is None:
            return
        cfg.ingest_rcfg([neighbour_id, 0, 0])

    def send_fission_command(self, robot_id: int) -> None:
        """Issue a fission command (split_command_code = -1)."""
        cfg = self.configurers.get(robot_id)
        if cfg is None:
            return
        cfg.ingest_rcfg([0, 0, -1])

    # ------------------------------------------------------------------
    #  Simulated pose update  (AMCL stand-in via xfm_vel integration)
    # ------------------------------------------------------------------

    def _integrate_pose(self, pose: Pose, xfm: Twist) -> Pose:
        """
        Simple Euler integration.  Robot frame velocities are interpreted
        in the robot's local frame and rotated into the world frame by yaw.
        """
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        dx_world = (c * xfm.linear_x - s * xfm.linear_y) * self.dt
        dy_world = (s * xfm.linear_x + c * xfm.linear_y) * self.dt
        dyaw     = xfm.angular_z * self.dt
        return Pose(
            x   = pose.x + dx_world,
            y   = pose.y + dy_world,
            yaw = pose.yaw + dyaw,
        )

    # ------------------------------------------------------------------
    #  Main step
    # ------------------------------------------------------------------

    def tick(self, distribute_cmd: Optional[Dict[int, Twist]] = None) -> None:
        """
        One simulation tick:
          1. Optionally push a fresh cmd_vel into each robot.
          2. For each robot, feed its own pose + any neighbour pose that
             appears in its rcfg, then step() the FSM.
          3. Integrate xfm_vel -> ground-truth pose.
        """
        self._step_ix += 1

        if distribute_cmd:
            for rid, cmd in distribute_cmd.items():
                self.send_cmd_vel(rid, cmd)

        # Feed poses in: own pose always; neighbour pose only if relevant.
        for rid, cfg in self.configurers.items():
            own = self.poses[rid]
            # Look up neighbour pose if fusion is pending or the far side
            # has announced itself via rcfg.
            neighbour_pose = None
            nid = cfg.rcfg[0]
            if nid != 0 and nid in self.poses:
                neighbour_pose = self.poses[nid]
            cfg.ingest_pose(R_h=own, R_n=neighbour_pose)

        # Step each Configurer
        fresh_xfm: Dict[int, Twist] = {}
        for rid, cfg in self.configurers.items():
            xfm, _state = cfg.step()
            fresh_xfm[rid] = xfm
            self._xfm_log[rid].append(xfm)

        # Integrate ground-truth poses
        for rid, xfm in fresh_xfm.items():
            self.poses[rid] = self._integrate_pose(self.poses[rid], xfm)

    # ------------------------------------------------------------------
    #  Visualisation  (fresh figure per call, blocking show -- Spyder-OK)
    # ------------------------------------------------------------------

    def viz(self, label: str = "") -> None:
        """
        Render current scene.  Fresh figure + plt.show() per call
        (no FuncAnimation).
        """
        if not self.visualize:
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        # 1. Draw fusion membership links (host -> each member at offset)
        by_host: Dict[int, List[int]] = {}
        for rid, cfg in self.configurers.items():
            by_host.setdefault(cfg.host_id, []).append(rid)
        for host_id, members in by_host.items():
            if len(members) < 2:
                continue
            # Draw a faint rectangle hull around the fused group
            xs = [self.poses[m].x for m in members]
            ys = [self.poses[m].y for m in members]
            pad = 0.4
            ax.add_patch(patches.Rectangle(
                (min(xs) - pad, min(ys) - pad),
                (max(xs) - min(xs)) + 2 * pad,
                (max(ys) - min(ys)) + 2 * pad,
                linewidth=2, edgecolor='#C1C1C1',
                facecolor='#EEEEEE', alpha=0.4, zorder=1,
            ))

        # 2. Draw each robot as a coloured square + heading arrow
        for rid, cfg in self.configurers.items():
            pose  = self.poses[rid]
            col   = self.COLORS[(rid - 1) % len(self.COLORS)]
            size  = 0.35

            # Body
            body = patches.Rectangle(
                (pose.x - size / 2, pose.y - size / 2), size, size,
                linewidth=1.5,
                edgecolor='black' if cfg.is_fused() else col,
                facecolor=col, alpha=0.85, zorder=3,
            )
            ax.add_patch(body)

            # Heading arrow
            hx = pose.x + 0.45 * math.cos(pose.yaw)
            hy = pose.y + 0.45 * math.sin(pose.yaw)
            ax.annotate(
                '', xy=(hx, hy), xytext=(pose.x, pose.y),
                arrowprops=dict(arrowstyle='->', color=col, lw=2),
                zorder=4,
            )

            # Label with id / state / n
            ax.text(
                pose.x, pose.y + 0.5,
                f"wm{rid}\n{cfg.fsm_state.value}  n={cfg.n}  host=wm{cfg.host_id}",
                ha='center', va='bottom', fontsize=8,
                color='black', zorder=5,
            )

        # 3. Auto-fit bounds with padding
        xs = [p.x for p in self.poses.values()]
        ys = [p.y for p in self.poses.values()]
        if xs and ys:
            pad = 1.5
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Configurer  |  {label}  |  step {self._step_ix}",
                     fontsize=12)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Traces (for tests / printed logs)
    # ------------------------------------------------------------------

    def xfm_log(self, robot_id: int) -> List[Twist]:
        return list(self._xfm_log.get(robot_id, []))

    def states_snapshot(self) -> Dict[int, Dict]:
        return {rid: cfg.snapshot() for rid, cfg in self.configurers.items()}


# ============================================================================
#  TEST HELPER:  synchronous two-party fusion (bypasses the SimBus loop)
# ============================================================================

def fuse_sync(a: Configurer, b: Configurer,
              pose_a: Pose, pose_b: Pose,
              max_iters: int = 10) -> None:
    """
    Atomic two-party fusion for unit tests / quick sanity checks.
    Wires temporary publish callbacks between a and b, seeds a fusion
    command into a, and drives both FSMs until they settle back to CONFIG.
    Restores the original publish callbacks on exit.
    """
    orig_a, orig_b = a._publish, b._publish

    def pub_from_a(tid, rcfg):
        if tid == b.robot_id:
            b.ingest_rcfg(rcfg)

    def pub_from_b(tid, rcfg):
        if tid == a.robot_id:
            a.ingest_rcfg(rcfg)

    a._publish = pub_from_a
    b._publish = pub_from_b
    try:
        a.ingest_pose(R_h=pose_a, R_n=pose_b)
        b.ingest_pose(R_h=pose_b, R_n=pose_a)
        a.ingest_rcfg([b.robot_id, 0, 0])     # user-initiated fusion

        for _ in range(max_iters):
            a.step()
            b.step()
            if (a.fsm_state == FSMState.CONFIG and
                    b.fsm_state == FSMState.CONFIG and
                    a.is_fused() and b.is_fused()):
                return
        raise RuntimeError("fuse_sync did not converge within max_iters.")
    finally:
        a._publish, b._publish = orig_a, orig_b


# ============================================================================
#  DEMO SCENARIO  (macOS Spyder-compatible)
# ============================================================================

def _make_bus(n_robots: int, visualize: bool) -> SimBus:
    """
    Build a SimBus with n_robots Configurers arranged in a line
    along the x-axis, spaced 1.5 m apart, all facing +x.
    """
    n_robots = max(2, min(5, n_robots))   # demo bounds: 2..5

    bus = SimBus(dt=0.2, visualize=visualize)
    for i in range(n_robots):
        rid = i + 1
        cfg = Configurer(
            robot_id=rid,
            on_publish_rcfg=bus.publish_rcfg,
            verbose=True,
        )
        bus.register(cfg, pose=Pose(x=i * 1.5, y=0.0, yaw=0.0))
    return bus


def _phase_cruise(bus: SimBus, ticks: int, label: str) -> None:
    """Run N ticks of uniform forward motion, push cmd_vel into every robot."""
    fwd = Twist(linear_x=0.3, angular_z=0.0)
    for _ in range(ticks):
        bus.tick(distribute_cmd={rid: fwd for rid in bus.configurers})
    bus.viz(label=label)


def _phase_rotate_fused(bus: SimBus, host_id: int, ticks: int, label: str) -> None:
    """
    Issue an angular cmd_vel to the host of a fused singleton.
    Members at nonzero rT_b offsets should produce xfm_vel with the
    d*sin(phi)*omega / -d*cos(phi)*omega correction terms -- this is the
    whole point of the framework and is visible in the xfm traces.
    """
    turn = Twist(linear_x=0.0, angular_z=0.4)
    for _ in range(ticks):
        # Only send cmd_vel to the host; in a real fused singleton the
        # nav stack publishes to the host namespace only.
        bus.tick(distribute_cmd={host_id: turn})
    bus.viz(label=label)


def _print_xfm_sample(bus: SimBus, k: int = 4) -> None:
    """Print the last k xfm_vel entries per robot for a quick trace."""
    print("\n--- last xfm_vel samples per robot ---")
    for rid in sorted(bus.configurers):
        tail = bus.xfm_log(rid)[-k:]
        print(f"  wm{rid}:")
        for i, t in enumerate(tail):
            print(f"    [{i}] vx={t.linear_x:+.3f}  vy={t.linear_y:+.3f}  "
                  f"wz={t.angular_z:+.3f}")


def demo(n_robots: int = 3, visualize: bool = True) -> None:
    """
    Scripted scenario (Spyder-runnable):
      1. Spawn N split singletons (SS), visualise initial state.
      2. Cruise forward -- verify SS robots act independently (rT_b = I).
      3. Fuse wm1 + wm2.  Visualise the fused singleton.
      4. Turn the fused singleton.  Trace how xfm_vel differs between
         host (zero offset) and members (omega-coupled offset terms).
      5. (If N>=3) Fuse wm3 into the formation, turn again.
      6. Fission all fused members.  Visualise SS again.
      7. Print state snapshots + tail of xfm traces.
    """
    bus = _make_bus(n_robots=n_robots, visualize=visualize)
    bus.viz(label="init (all SS)")

    _phase_cruise(bus, ticks=5, label="SS cruise")

    # --- Fuse wm1 + wm2 (handshake takes a couple of ticks) ---
    print("\n>>> FUSE  wm1 <- wm2")
    bus.send_fusion_command(host_id=1, neighbour_id=2)
    bus.send_fusion_command(host_id=2, neighbour_id=1)
    # Drive ticks until both settle -- bounded loop is safe because
    # the sample protocol completes in 2 ticks.
    for _ in range(6):
        bus.tick()
        if all(c.fsm_state == FSMState.CONFIG for c in bus.configurers.values()):
            break
    bus.viz(label="after fuse wm1+wm2")

    _phase_rotate_fused(bus, host_id=1, ticks=5, label="fused wm1+wm2 turning")

    # --- If more than two, fuse wm3 in next ---
    if n_robots >= 3:
        print("\n>>> FUSE  wm1 <- wm3")
        bus.send_fusion_command(host_id=1, neighbour_id=3)
        bus.send_fusion_command(host_id=3, neighbour_id=1)
        for _ in range(6):
            bus.tick()
            if all(c.fsm_state == FSMState.CONFIG for c in bus.configurers.values()):
                break
        bus.viz(label="after fuse wm1+wm2+wm3")
        _phase_rotate_fused(bus, host_id=1, ticks=4, label="3-robot FS turning")

    # --- Fission: split every fused member back to SS ---
    print("\n>>> FISSION all fused members")
    for rid, cfg in bus.configurers.items():
        if cfg.is_fused():
            bus.send_fission_command(rid)
    for _ in range(3):
        bus.tick()
        if all(c.is_split_singleton() for c in bus.configurers.values()):
            break
    bus.viz(label="after fission (all SS again)")

    _phase_cruise(bus, ticks=3, label="SS cruise after fission")

    # --- Final state printout ---
    print("\n--- final states ---")
    for rid, snap in bus.states_snapshot().items():
        print(f"  wm{rid}:  {snap}")
    _print_xfm_sample(bus, k=3)


# ============================================================================
#  PYGAME TELEOP  (lazy-loaded:  pygame only imported when this file is run)
# ============================================================================
#
# All pygame code (constants, helpers, TeleopSim class, main loop) lives
# inside _run_pygame_teleop().  Nothing here is evaluated when Configurer.py
# is imported from another script, so the library surface remains pygame-free
# and ROS2-compatible.
#
# Controls:
#   1..5                 select robot
#   W / S  or  ↑ / ↓     forward / reverse
#   A / D                rotate left / right       (differential / yaw)
#   ← / →                strafe left / right       (holonomic)
#   Q / E                scale velocity up / down   (Q+E = emergency stop)
#   SHIFT + N            fuse selected with wmN    (iff neighbours)
#   SPACE                fission formation containing selected
#   T                    attach/detach trolley (nearest in range)
#   R                    reset scenario
#   ESC                  quit
#   LMB-click            point-to-point goal (A* / Dijkstra)
#   SHIFT (held)         reveal occupancy overlay
# ============================================================================

def _run_pygame_teleop() -> int:
    """
    Interactive pygame teleop sandbox.  Entry point for `python Configurer.py`.

    Every pygame-touching symbol (constants, helpers, TeleopSim class,
    imports) is defined inside this function so that importing Configurer
    from another script never triggers pygame / SDL initialisation.

    Sim-level model (beyond the Configurer FSM)
    -------------------------------------------
    * Each robot belongs to a FORMATION, tracked by `self.formation_of[rid]`.
      This is the authoritative grouping (not cfg.host_id, which can be
      stale in multi-formation chain fusions).  Teleop cmd_vel routing,
      collision grouping, and the visual hull all use formation_of.
    * Each formation has a canonical HOST = min(rid in formation).
      Only the host accepts teleop cmd_vel (non-host members sit idle).
    * DOCKING: SHIFT+N first checks the closest-pair distance between the
      two formations is within the DOCKING radius.  If so, the triggering
      formation's host drives toward the target formation's host under
      sim control, rotating to align yaw.  When within DOCKED distance
      and yaw tolerance, the formations unify (rT_b/host_id/n recomputed
      from live poses) and teleop resumes.
    * COLLISION: robots in different formations cannot occupy the same
      space -- after each bus tick, any overlap within DOCKED_DISTANCE
      is resolved by pushing both formations apart along the normal.
      The pair currently docking is exempt (otherwise they'd never touch).
    """
    try:
        import pygame
    except ImportError:
        print("pygame is not installed.  Install with:  pip install pygame",
              file=sys.stderr)
        return 2

    from common.obstacles import (
        ObstacleKind, ObstacleManager, Obstacle,
        OBSTACLE_LABELS, OBSTACLE_DIMS, CASTER_RADIUS,
        TROLLEY_WEIGHT_CLASS, _CASTER_TROLLEYS, _DRAG_SPAWN,
        snap_angle_45,
        pathfind_astar, pathfind_dijkstra, smooth_path,
    )

    # ------------------------------------------------------------------
    #  Window / canvas
    # ------------------------------------------------------------------
    WINDOW_W       = 1100
    WINDOW_H       = 750
    HUD_H          = 130                      # reserved strip at top
    TOOLBAR_W      = 130                      # right-side toolbar for obstacles
    FPS            = 30

    WORLD_W        = 22.0                     # world metres across canvas
    WORLD_H        = 15.0

    # ------------------------------------------------------------------
    #  Robots + occupancy + docking geometry
    #
    #  ROBOT_OCCUPANCY_M  : effective body radius for collision
    #  DOCKED_DISTANCE_M  : centre-to-centre distance when two robots are
    #                      touching (= 2 * occupancy radius)
    #  DOCKING_DISTANCE_M : centre-to-centre distance at which fusion is
    #                      eligible (> docked distance)
    #  docking_range_m   := DOCKING_DISTANCE_M - DOCKED_DISTANCE_M
    # ------------------------------------------------------------------
    N_ROBOTS             = 2                  # standalone demo: 2 robots, FSM + A* + GBNN only
    ROBOT_SIZE_M         = 0.35               # body half-width (visual)
    ROBOT_OCCUPANCY_M    = 0.40               # collision radius per robot
    DOCKED_DISTANCE_M    = 2.0 * ROBOT_OCCUPANCY_M    # = 0.80 m
    DOCKING_DISTANCE_M   = 2.00                       # eligibility radius
    INITIAL_SPACING_M    = DOCKING_DISTANCE_M * 0.75  # = 1.50 m (inside range)

    DT                   = 1.0 / FPS
    BASE_LIN_SPEED       = 0.60               # m/s, scaled by vel_scale
    BASE_ANG_SPEED       = 1.20               # rad/s
    VEL_SCALE_MIN        = 0.10
    VEL_SCALE_MAX        = 3.00
    VEL_SCALE_STEP       = 0.03               # per-tick while Q or E held

    # Docking controller
    DOCK_LIN_GAIN        = 1.8                # P-gain on translation error
    DOCK_ANG_GAIN        = 3.0                # P-gain on yaw error
    DOCK_LIN_MAX         = 1.0                # m/s cap
    DOCK_ANG_MAX         = 2.5                # rad/s cap
    DOCK_POS_TOL_M       = 0.04               # snap when within this
    DOCK_YAW_TOL_RAD     = 0.05
    DOCK_TIMEOUT_TICKS   = 300                # ~10 sec at 30 FPS

    # Trolley attachment
    TROLLEY_ATTACH_RANGE_M = 1.5              # must be within this to initiate
    TROLLEY_DOCK_POS_TOL   = 0.05             # snap tolerance (m)
    TROLLEY_DOCK_YAW_TOL   = 0.08             # snap tolerance (rad)
    TROLLEY_EDGE_OFFSET    = 0.05             # offset robot edge ↔ trolley edge

    # Pathfinding
    COLLISION_EPSILON    = 0.002              # ignore overlaps < 2 mm (prevents micro-bounce jitter)

    # Pathfinding
    PATHFIND_CELL_SIZE   = 0.20               # occupancy grid resolution (m)
    PATHFIND_INFLATE     = 0.35               # inflate obstacles (< ROBOT_OCCUPANCY_M to avoid self-block)
    PATHFIND_WAYPOINT_TOL = 0.15              # snap to next waypoint (m)
    PATHFIND_REPLAN_INTERVAL = 10             # recompute path every N ticks (not every frame)
    PATHFIND_FAIL_LIMIT  = 60                 # cancel nav after this many consecutive failures (~2s)

    # ------------------------------------------------------------------
    #  Palette
    # ------------------------------------------------------------------
    BG_COLOUR          = (18, 18, 24)
    GRID_COLOUR        = (40, 40, 52)
    HUD_BG             = (28, 28, 36)
    HUD_TEXT           = (220, 220, 230)
    HUD_ACCENT         = (255, 190, 60)
    HUD_WARN           = (240, 90, 90)
    HUD_OK             = (90, 220, 140)
    HULL_EDGE          = (120, 120, 140)
    HULL_FILL          = (55, 55, 72)
    SELECTED_RING      = (255, 225, 90)
    DOCK_RING          = (110, 150, 200)      # pale blue docking radius
    DOCK_ACTIVE_RING   = (170, 210, 255)      # brighter while docking
    ROT_CENTRE_COL     = (255, 200, 60)       # rotation centre crosshair
    ROT_CENTRE_COL_C   = (255, 200, 60)       # centroid mode
    ROT_CENTRE_COL_H   = (255, 120, 60)       # host mode (orange-ish)
    TOOLBAR_BG         = (30, 30, 40)
    TOOLBAR_BTN        = (50, 50, 65)
    TOOLBAR_BTN_SEL    = (80, 120, 180)
    TOOLBAR_TEXT       = (200, 200, 210)
    PATH_LINE_COL      = (255, 255, 80, 160) # yellow path
    PATH_GOAL_COL      = (255, 80, 80)       # red goal marker
    ROBOT_COLOURS      = [
        (66, 135, 245),  # 1 blue
        (80, 200, 120),  # 2 green
        (235, 85, 75),   # 3 red
        (80, 200, 210),  # 4 cyan
        (205, 110, 220), # 5 magenta
        (255, 165, 40),  # 6 orange (reserve)
    ]

    # ------------------------------------------------------------------
    #  Input snapshot dataclass
    # ------------------------------------------------------------------
    @dataclass
    class InputState:
        w: bool = False
        a: bool = False
        s: bool = False
        d: bool = False
        up: bool = False        # ↑  — forward (alias for W)
        down: bool = False      # ↓  — reverse (alias for S)
        left: bool = False      # ←  — yaw left
        right: bool = False     # →  — yaw right
        shift: bool = False
        q: bool = False
        e: bool = False

    # ------------------------------------------------------------------
    #  TeleopSim
    # ------------------------------------------------------------------
    class TeleopSim:

        def __init__(self) -> None:
            pygame.init()
            pygame.display.set_caption(
                "Configurer Teleop -- WASD drive | "
                "SHIFT+A/D strafe | SHIFT+N dock | SPACE fission | T trolley"
            )
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            self.clock  = pygame.time.Clock()

            self.font_small = pygame.font.SysFont("menlo", 14)
            self.font_med   = pygame.font.SysFont("menlo", 16, bold=True)
            self.font_big   = pygame.font.SysFont("menlo", 22, bold=True)

            self.bus:          SimBus               = None   # type: ignore
            self.selected_id:  int                  = 1
            self.vel_scale:    float                = 1.0
            self.last_message: str                  = (
                "Ready.  Press 1-5 to select a robot."
            )
            self.last_msg_col: Tuple[int, int, int] = HUD_TEXT
            self.input_state:  InputState           = InputState()
            self.running:      bool                 = True
            self._last_cmd:    Twist                = Twist()

            # Sim-level grouping + docking state
            self.formation_of: Dict[int, int]   = {}
            self.docking:      Optional[Dict]   = None

            # Rotation-centre mode: "centroid" (paper Eq 6-7) or "host"
            self.rot_centre_mode: str = "centroid"

            # Centroid anchor: tracks ideal centroid position per formation
            # to eliminate Euler orbital drift.  Key = fid, value = (x, y).
            self._centroid_anchor: Dict[int, Tuple[float, float]] = {}

            # ---- Obstacle system ----
            self.obs_mgr = ObstacleManager()
            self._toolbar_selected: Optional[ObstacleKind] = None
            self._toolbar_kinds: List[ObstacleKind] = list(ObstacleKind)

            # ---- Mouse drag state ----
            self._mouse_down_pos: Optional[Tuple[int, int]] = None
            self._mouse_down_time: float = 0.0
            self._mouse_held:     bool   = False
            self._last_click_time: float  = 0.0
            self._DOUBLE_CLICK_MS: int    = 400

            # ---- Robot drag state ----
            self._dragging_robot: Optional[int] = None   # robot id being dragged
            self._drag_robot_offset: Tuple[float, float] = (0.0, 0.0)  # offset from click to robot centre

            # ---- Drag-spawn state (walls, doors) ----
            self._drag_spawn_start: Optional[Tuple[float, float]] = None  # world coords
            self._drag_spawn_preview: Optional[dict] = None  # {x, y, yaw, half_w, kind}


            # ---- Pathfinding state ----
            self._pathfind_algo: str = "astar"   # "astar" or "dijkstra"
            self._nav_motion: str = "holonomic"  # "differential" / "holonomic" / "hybrid"
            self._nav_goals: Dict[int, Tuple[float, float]] = {}  # rid -> goal
            self._nav_paths: Dict[int, List[Tuple[float, float]]] = {}
            self._nav_wp_idx: Dict[int, int] = {}  # current waypoint index
            self._nav_replan_tick: Dict[int, int] = {}  # tick counter for replan
            self._nav_fail_count: Dict[int, int] = {}   # consecutive path failures

            # Contact normals from last-frame obstacle collisions.
            # Maps rid -> list of (nx, ny) unit normals pointing AWAY
            # from the obstacle surface.  Used to zero-out velocity
            # components that would push the robot back into the wall.
            self._contact_normals: Dict[int, List[Tuple[float, float]]] = {}

            # ---- Trolley attachment state ----
            # Maps the SPECIFIC robot_rid that physically attached ->
            # trolley obstacle uid.  This rid does NOT change when
            # formations merge/split — it always tracks the original
            # attaching robot.
            self._attached_trolley: Dict[int, int] = {}
            # Active attachment docking sequence (only one at a time)
            self._trolley_docking: Optional[Dict] = None

            self._spawn_scenario()

        # ----- scenario --------------------------------------------------
        def _spawn_scenario(self) -> None:
            self.bus = SimBus(dt=DT, visualize=False)  # no matplotlib here
            y0 = 0.0
            x0 = -INITIAL_SPACING_M * (N_ROBOTS - 1) / 2.0
            for i in range(N_ROBOTS):
                rid = i + 1
                cfg = Configurer(
                    robot_id=rid,
                    on_publish_rcfg=self.bus.publish_rcfg,
                    verbose=False,
                )
                pose = Pose(x=x0 + i * INITIAL_SPACING_M, y=y0, yaw=0.0)
                self.bus.register(cfg, pose)

            self.selected_id   = 1
            self.vel_scale     = 1.0
            self.docking       = None
            self.formation_of  = {rid: rid for rid in self.bus.configurers}
            self.obs_mgr.obstacles.clear()
            self.obs_mgr.dragging_id = None
            self._nav_goals.clear()
            self._nav_paths.clear()
            self._nav_wp_idx.clear()
            self._nav_replan_tick.clear()
            self._nav_fail_count.clear()
            self._centroid_anchor.clear()
            self._attached_trolley.clear()
            self._trolley_docking = None
            self._set_message(
                f"Scenario reset.  {N_ROBOTS} split singletons ready.",
                HUD_OK,
            )

        def _set_message(self,
                         text: str,
                         colour: Tuple[int, int, int] = HUD_TEXT) -> None:
            self.last_message = text
            self.last_msg_col = colour

        # ----- sim-level formation helpers -------------------------------
        def _members_of(self, rid: int) -> List[int]:
            fid = self.formation_of[rid]
            return [r for r, f in self.formation_of.items() if f == fid]

        def _host_of(self, rid: int) -> int:
            return min(self._members_of(rid))

        def _formations_by_fid(self) -> Dict[int, List[int]]:
            out: Dict[int, List[int]] = {}
            for r, f in self.formation_of.items():
                out.setdefault(f, []).append(r)
            for fid in out:
                out[fid].sort()
            return out

        def _closest_pair_distance(self,
                                   ids_a: List[int],
                                   ids_b: List[int]) -> float:
            best = math.inf
            for a in ids_a:
                pa = self.bus.poses.get(a)
                if pa is None:
                    continue
                for b in ids_b:
                    pb = self.bus.poses.get(b)
                    if pb is None:
                        continue
                    d = math.hypot(pa.x - pb.x, pa.y - pb.y)
                    if d < best:
                        best = d
            return best

        def _formation_centroid(self, rid: int) -> Tuple[float, float]:
            """
            Weighted centroid of the formation containing *rid* in
            WORLD FRAME (from live poses).  Used for rendering only.
            For velocity compensation use _centroid_offset_body().
            """
            members = self._members_of(rid)
            cx = sum(self.bus.poses[m].x for m in members) / len(members)
            cy = sum(self.bus.poses[m].y for m in members) / len(members)
            return cx, cy

        def _centroid_offset_body(self, rid: int) -> Tuple[float, float]:
            """
            Rotation-centre offset from the host, in the HOST's body frame.

            Normal behaviour (no HEAVY_TROLLEY):
                Returns the geometric centroid of all formation members.

            HEAVY_TROLLEY override:
                When a member has a HEAVY_TROLLEY attached, the rotation
                centre is pinned to that member (the trolley centroid).
                Returns that member's body-frame offset from host.

            Math
            ----
            rT_b[:2,2] stores  R_own^T @ (-(p_own - p_host)).
            At dock time all yaws are aligned, so R_own = R_host, giving:
              rT_b[:2,2] = -(p_own - p_host) in host body frame.

            The offset of member m from host in host body frame is:
              offset_m = -rT_b_m[:2,2]

            Centroid in host body frame:
              C_body = (1/n) * sum(offset_m)  for all members
                     = -(1/n) * sum(rT_b_m[:2,2])
            """
            # Check for HEAVY_TROLLEY override
            att_rid = self._formation_trolley_robot(rid)
            if att_rid is not None:
                uid = self._attached_trolley[att_rid]
                obs = self.obs_mgr.obstacles.get(uid)
                if obs and obs.kind == ObstacleKind.HEAVY_TROLLEY:
                    # Return the attached robot's offset from host
                    rTb = self.bus.configurers[att_rid].rT_b
                    return -rTb[0, 2], -rTb[1, 2]

            # Normal: geometric centroid
            members = self._members_of(rid)
            n = len(members)
            sx = sy = 0.0
            for m in members:
                rTb = self.bus.configurers[m].rT_b
                sx += rTb[0, 2]
                sy += rTb[1, 2]
            return -sx / n, -sy / n

        def _rotation_centre(self, rid: int) -> Tuple[float, float]:
            """Return the current rotation centre for *rid*'s formation
            in WORLD FRAME (for rendering the crosshair)."""
            if self.rot_centre_mode == "centroid":
                # HEAVY_TROLLEY override: pinned to attached robot
                att_rid = self._formation_trolley_robot(rid)
                if att_rid is not None:
                    uid = self._attached_trolley[att_rid]
                    obs = self.obs_mgr.obstacles.get(uid)
                    if obs and obs.kind == ObstacleKind.HEAVY_TROLLEY:
                        pa = self.bus.poses[att_rid]
                        return pa.x, pa.y
                return self._formation_centroid(rid)
            else:  # "host"
                host = self._host_of(rid)
                ph = self.bus.poses[host]
                return ph.x, ph.y

        # ----- world <-> screen mapping ----------------------------------
        def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
            canvas_h = WINDOW_H - HUD_H
            px_per_m = min(WINDOW_W / WORLD_W, canvas_h / WORLD_H)
            cx = WINDOW_W / 2.0
            cy = HUD_H + canvas_h / 2.0
            sx = cx + x * px_per_m
            sy = cy - y * px_per_m
            return int(sx), int(sy)

        def _metres_to_px(self, m: float) -> int:
            canvas_h = WINDOW_H - HUD_H
            return int(m * min(WINDOW_W / WORLD_W, canvas_h / WORLD_H))

        def _screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
            """Inverse of _world_to_screen."""
            canvas_h = WINDOW_H - HUD_H
            px_per_m = min(WINDOW_W / WORLD_W, canvas_h / WORLD_H)
            cx = WINDOW_W / 2.0
            cy = HUD_H + canvas_h / 2.0
            wx = (sx - cx) / px_per_m
            wy = (cy - sy) / px_per_m
            return wx, wy

        def _is_in_toolbar(self, sx: int, sy: int) -> bool:
            """Check if screen coords fall inside the right toolbar."""
            return sx >= WINDOW_W - TOOLBAR_W and sy >= HUD_H

        def _toolbar_kind_at(self, sx: int, sy: int) -> Optional[ObstacleKind]:
            """Return the ObstacleKind for a click position in the toolbar."""
            if not self._is_in_toolbar(sx, sy):
                return None
            btn_h = 32
            gap = 4
            y_off = sy - (HUD_H + 40)   # skip header area
            if y_off < 0:
                return None
            idx = y_off // (btn_h + gap)
            if 0 <= idx < len(self._toolbar_kinds):
                return self._toolbar_kinds[idx]
            return None

        def _world_bounds(self) -> Tuple[float, float, float, float]:
            return (-WORLD_W / 2, -WORLD_H / 2, WORLD_W / 2, WORLD_H / 2)

        def _robot_at(self, wx: float, wy: float) -> Optional[int]:
            """Return robot id whose centre is within ROBOT_OCCUPANCY_M of (wx, wy), or None."""
            best_id: Optional[int] = None
            best_d2 = float("inf")
            for rid in self.bus.configurers:
                p = self.bus.poses[rid]
                d2 = (p.x - wx) ** 2 + (p.y - wy) ** 2
                if d2 < ROBOT_OCCUPANCY_M ** 2 and d2 < best_d2:
                    best_id = rid
                    best_d2 = d2
            return best_id

        # ----- mouse events ----------------------------------------------

        def _on_mousedown(self, event) -> None:
            sx, sy = event.pos
            now_ms = pygame.time.get_ticks()

            # Toolbar click: select obstacle type to spawn
            kind = self._toolbar_kind_at(sx, sy)
            if kind is not None:
                if self._toolbar_selected == kind:
                    self._toolbar_selected = None
                    self._set_message("Obstacle deselected.", HUD_TEXT)
                else:
                    self._toolbar_selected = kind
                    if kind in _DRAG_SPAWN:
                        self._set_message(
                            f"Selected: {OBSTACLE_LABELS[kind]} — "
                            f"click & drag to spawn, dbl-click to despawn.",
                            HUD_ACCENT)
                    else:
                        self._set_message(
                            f"Selected: {OBSTACLE_LABELS[kind]} — "
                            f"click canvas to spawn, dbl-click to despawn.",
                        HUD_ACCENT)
                return

            # Ignore clicks in HUD area
            if sy < HUD_H:
                return

            wx, wy = self._screen_to_world(sx, sy)

            # ── Double-click detection: despawn obstacle ──
            # Must be checked HERE (mousedown) because mousedown on
            # an obstacle starts a drag, and mouseup returns early
            # after ending the drag — so mouseup never reaches the
            # double-click check.
            is_dbl = (now_ms - self._last_click_time < self._DOUBLE_CLICK_MS)
            self._last_click_time = now_ms

            if is_dbl:
                hit = self.obs_mgr.obstacle_at(wx, wy)
                if hit is not None:
                    self.obs_mgr.despawn(hit.uid)
                    self._set_message(
                        f"Despawned {hit.label}.", HUD_WARN)
                    self._last_click_time = 0   # reset to prevent triple
                    return

            # Record mouse down for drag / single-click detection
            self._mouse_down_pos = (sx, sy)
            self._mouse_down_time = now_ms
            self._mouse_held = False

            # Start drag-spawn for wall / door types
            if (self._toolbar_selected is not None
                    and self._toolbar_selected in _DRAG_SPAWN
                    and not self._is_in_toolbar(sx, sy)
                    and sy >= HUD_H):
                self._drag_spawn_start = (wx, wy)
                self._drag_spawn_preview = None
                self._mouse_held = True
                return

            # Check if clicking on an existing obstacle (start drag)
            hit_obs = self.obs_mgr.obstacle_at(wx, wy)
            if hit_obs is not None:
                self.obs_mgr.start_drag(hit_obs.uid, wx, wy)
                self._mouse_held = True
                return

            # Check if clicking on a robot (start robot drag)
            if self._toolbar_selected is None:
                hit_rid = self._robot_at(wx, wy)
                if hit_rid is not None:
                    rp = self.bus.poses[hit_rid]
                    self._dragging_robot = hit_rid
                    self._drag_robot_offset = (wx - rp.x, wy - rp.y)
                    # Cancel nav for this formation
                    host = self._host_of(hit_rid)
                    self._cancel_nav(host)
                    self._mouse_held = True
                    return

        def _on_mouseup(self, event) -> None:
            sx, sy = event.pos

            # End obstacle drag if active
            if self.obs_mgr.dragging_id is not None:
                self.obs_mgr.end_drag()
                self._mouse_down_pos = None
                return

            # End robot drag if active
            if self._dragging_robot is not None:
                self._dragging_robot = None
                self._mouse_down_pos = None
                return

            # Finalize drag-spawn (walls, doors)
            if self._drag_spawn_start is not None:
                preview = self._drag_spawn_preview
                self._drag_spawn_start = None
                self._drag_spawn_preview = None
                if preview is not None:
                    hw = preview["half_w"]
                    hh = preview.get("half_h")
                    kind = preview["kind"]
                    # Only spawn wall if drag was long enough
                    if kind == ObstacleKind.WALL and hw < 0.2:
                        self._set_message(
                            "Drag too short to spawn a wall.", HUD_WARN)
                        self._mouse_down_pos = None
                        return
                    # Only spawn pillar if drag was long enough
                    if kind == ObstacleKind.PILLAR and hw < 0.10:
                        self._set_message(
                            "Drag too short to spawn a pillar.", HUD_WARN)
                        self._mouse_down_pos = None
                        return
                    # Determine custom size overrides
                    spawn_hw = None
                    spawn_hh = None
                    if kind == ObstacleKind.WALL:
                        spawn_hw = hw
                    elif kind == ObstacleKind.PILLAR:
                        spawn_hw = hw   # radius
                        spawn_hh = hh   # same radius (round)
                    obs = self.obs_mgr.spawn(
                        kind, preview["x"], preview["y"],
                        yaw=preview["yaw"],
                        half_w=spawn_hw, half_h=spawn_hh)
                    self._set_message(
                        f"Spawned {obs.label} at "
                        f"({preview['x']:.1f}, {preview['y']:.1f}).",
                        HUD_OK)
                self._mouse_down_pos = None
                return

            if self._mouse_down_pos is None:
                return

            wx, wy = self._screen_to_world(sx, sy)

            # Single click on canvas: spawn if toolbar selected
            # (only for non-drag-spawn types like table, chair, etc.)
            if self._toolbar_selected is not None:
                if (self._toolbar_selected not in _DRAG_SPAWN
                        and not self._is_in_toolbar(sx, sy)
                        and sy >= HUD_H):
                    obs = self.obs_mgr.spawn(self._toolbar_selected, wx, wy)
                    self._set_message(
                        f"Spawned {obs.label} at ({wx:.1f}, {wy:.1f}).",
                        HUD_OK)
                    self._mouse_down_pos = None
                    return

            # Single click on free space with no toolbar: pathfind goal
            # Only the HOST of a formation can receive nav goals.
            # Clicking while a non-host member is selected is ignored.
            if (self._toolbar_selected is None
                    and self.obs_mgr.obstacle_at(wx, wy) is None
                    and sy >= HUD_H
                    and not self._is_in_toolbar(sx, sy)):
                sel = self.selected_id
                if sel in self.bus.configurers:
                    host = self._host_of(sel)
                    members = self._members_of(sel)
                    if sel != host and len(members) > 1:
                        self._set_message(
                            f"wm{sel} is not the host. Select wm{host} "
                            f"to set a navigation goal.",
                            HUD_WARN)
                    else:
                        self._nav_goals[host] = (wx, wy)
                        if len(members) > 1:
                            tag = (f"formation (host=wm{host}, "
                                   f"n={len(members)})")
                        else:
                            tag = f"wm{host}"
                        self._set_message(
                            f"{tag} goal → ({wx:.1f}, {wy:.1f}) "
                            f"[{self._pathfind_algo.upper()} / "
                            f"{self._nav_motion.upper()}]",
                            HUD_ACCENT)

            self._mouse_down_pos = None

        def _on_mousemotion(self, event) -> None:
            if self.obs_mgr.dragging_id is not None:
                wx, wy = self._screen_to_world(*event.pos)
                self.obs_mgr.update_drag(wx, wy)
                return

            # Robot drag: move entire formation
            if self._dragging_robot is not None:
                wx, wy = self._screen_to_world(*event.pos)
                ox, oy = self._drag_robot_offset
                target_x = wx - ox
                target_y = wy - oy

                rid = self._dragging_robot
                host = self._host_of(rid)
                members = self._members_of(rid)

                # Compute delta from host's current position
                hp = self.bus.poses[host]
                dx = target_x - self.bus.poses[rid].x
                dy = target_y - self.bus.poses[rid].y

                # Shift entire formation by the same delta
                for m in members:
                    p = self.bus.poses[m]
                    self.bus.poses[m] = Pose(
                        x=p.x + dx, y=p.y + dy, yaw=p.yaw)

                # Update centroid anchor so pin_centroid doesn't snap back
                fid = self.formation_of.get(host)
                if fid is not None and fid in self._centroid_anchor:
                    ca_x, ca_y = self._centroid_anchor[fid]
                    self._centroid_anchor[fid] = (ca_x + dx, ca_y + dy)
                return

            # Update drag-spawn preview
            if self._drag_spawn_start is not None:
                wx, wy = self._screen_to_world(*event.pos)
                sx, sy = self._drag_spawn_start
                dx, dy = wx - sx, wy - sy
                drag_len = math.hypot(dx, dy)
                raw_angle = math.atan2(dy, dx)
                snapped_yaw = snap_angle_45(raw_angle)
                kind = self._toolbar_selected

                # Midpoint of the drag = centre of the obstacle
                mid_x = (sx + wx) / 2.0
                mid_y = (sy + wy) / 2.0

                if kind == ObstacleKind.WALL:
                    # Variable length wall: half_w = half of drag distance
                    half_w = max(0.2, drag_len / 2.0)
                    half_h = OBSTACLE_DIMS[kind][1]
                elif kind == ObstacleKind.PILLAR:
                    # Variable radius pillar: drag distance = radius
                    radius = max(0.10, drag_len)
                    half_w = radius
                    half_h = radius
                    # Centre at start of drag (click position)
                    mid_x, mid_y = sx, sy
                    snapped_yaw = 0.0  # pillar is round, yaw irrelevant
                else:
                    # Doors: fixed length, just orientation
                    half_w = OBSTACLE_DIMS[kind][0]
                    half_h = OBSTACLE_DIMS[kind][1]
                    # Centre at start of drag (not midpoint) for doors
                    mid_x, mid_y = sx, sy

                self._drag_spawn_preview = {
                    "x": mid_x, "y": mid_y,
                    "yaw": snapped_yaw,
                    "half_w": half_w,
                    "half_h": half_h,
                    "kind": kind,
                }

        # ----- events ----------------------------------------------------
        def _on_keydown(self, event) -> None:
            key   = event.key
            mods  = pygame.key.get_mods()
            shift = bool(mods & pygame.KMOD_SHIFT)

            digit_map = {
                pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
                pygame.K_4: 4, pygame.K_5: 5,
            }
            if key in digit_map:
                target = digit_map[key]
                if target > N_ROBOTS:
                    return
                if shift:
                    self._try_fuse(self.selected_id, target)
                else:
                    self._select_robot(target)
                return

            if key == pygame.K_SPACE:
                self._try_fission(self.selected_id)
                return
            if key == pygame.K_z:
                # Toggle rotation centre between centroid and host
                if self.rot_centre_mode == "centroid":
                    self.rot_centre_mode = "host"
                    self._set_message(
                        "Rotation centre: HOST robot position.",
                        HUD_ACCENT,
                    )
                else:
                    self.rot_centre_mode = "centroid"
                    self._set_message(
                        "Rotation centre: computed CENTROID (paper Eq 6-7).",
                        HUD_ACCENT,
                    )
                return
            if key == pygame.K_r:
                self._spawn_scenario()
                return
            if key == pygame.K_p:
                # Toggle pathfinding algorithm
                if self._pathfind_algo == "astar":
                    self._pathfind_algo = "dijkstra"
                else:
                    self._pathfind_algo = "astar"
                self._set_message(
                    f"Pathfinding algo: {self._pathfind_algo.upper()}  "
                    f"motion: {self._nav_motion.upper()}",
                    HUD_ACCENT)
                return
            if key == pygame.K_m:
                # Cycle motion mode: differential -> holonomic -> hybrid
                cycle = ["differential", "holonomic", "hybrid"]
                idx = cycle.index(self._nav_motion)
                self._nav_motion = cycle[(idx + 1) % len(cycle)]
                self._set_message(
                    f"Nav motion: {self._nav_motion.upper()}  "
                    f"algo: {self._pathfind_algo.upper()}",
                    HUD_ACCENT)
                return
            if key == pygame.K_x:
                # Cancel navigation for selected robot's formation
                sel = self.selected_id
                host = self._host_of(sel) if sel in self.bus.configurers else sel
                self._cancel_nav(host, f"wm{host} navigation cancelled.")
                return
            if key == pygame.K_t:
                self._try_trolley_toggle()
                return
            if key == pygame.K_c:
                # Clear all obstacles
                self.obs_mgr.obstacles.clear()
                self._attached_trolley.clear()
                self._trolley_docking = None
                self._set_message("All obstacles cleared.", HUD_WARN)
                return
            if key == pygame.K_ESCAPE:
                self.running = False
                return

        def _sync_keyboard_state(self) -> None:
            keys = pygame.key.get_pressed()
            s = self.input_state
            s.w = keys[pygame.K_w]
            s.a = keys[pygame.K_a]
            s.s = keys[pygame.K_s]
            s.d = keys[pygame.K_d]
            s.up    = keys[pygame.K_UP]
            s.down  = keys[pygame.K_DOWN]
            s.left  = keys[pygame.K_LEFT]
            s.right = keys[pygame.K_RIGHT]
            s.shift = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            s.q = keys[pygame.K_q]
            s.e = keys[pygame.K_e]

        # ----- teleop actions --------------------------------------------
        def _select_robot(self, rid: int) -> None:
            if rid not in self.bus.configurers:
                return
            self.selected_id = rid
            members = self._members_of(rid)
            host    = self._host_of(rid)
            # Label host with [H] when HEAVY_TROLLEY is attached
            att_rid = self._formation_trolley_robot(rid)
            heavy_tag = ""
            if att_rid is not None:
                uid = self._attached_trolley[att_rid]
                obs = self.obs_mgr.obstacles.get(uid)
                if obs and obs.kind == ObstacleKind.HEAVY_TROLLEY:
                    heavy_tag = "[H]"
            host_label = f"wm{host}{heavy_tag}"
            if len(members) > 1:
                role = "HOST" if rid == host else "member"
                hint = ("drives formation"
                        if rid == host else
                        f"idle (host is {host_label})")
                self._set_message(
                    f"Controlling wm{rid} [{role}, {hint}]  "
                    f"formation={sorted(members)} n={len(members)}",
                    HUD_ACCENT,
                )
            else:
                tag = f"{heavy_tag} " if heavy_tag else ""
                self._set_message(
                    f"Controlling wm{rid} {tag}(split singleton).",
                    HUD_ACCENT,
                )

        def _try_fuse(self, rid_a: int, rid_b: int) -> None:
            """
            Initiate docking between the formation of *rid_a* (trigger)
            and the formation of *rid_b* (target).

            Physical docking target
            -----------------------
            The trigger formation's host drives toward **rid_b** (the
            specific robot the user pointed at), NOT toward the target
            formation's host.  This is the key distinction:

              * dock_target  = rid_b        (who we physically approach)
              * new host     = min(all)     (formation naming, computed
                                             later in _complete_dock)

            After docking, the unified formation adopts min(all members)
            as its host -- that's a separate concern from the approach
            geometry.
            """
            if self.docking is not None or self._trolley_docking is not None:
                self._set_message(
                    "Docking/attaching in progress. Wait for it to complete.",
                    HUD_WARN,
                )
                return
            if rid_a == rid_b:
                self._set_message("Cannot fuse a robot with itself.",
                                  HUD_WARN)
                return
            if (rid_a not in self.bus.configurers
                    or rid_b not in self.bus.configurers):
                self._set_message("Unknown robot in fuse request.",
                                  HUD_WARN)
                return

            fid_a = self.formation_of[rid_a]
            fid_b = self.formation_of[rid_b]
            if fid_a == fid_b:
                self._set_message(
                    f"wm{rid_a} and wm{rid_b} already in the same formation.",
                    HUD_WARN,
                )
                return

            members_a = [r for r, f in self.formation_of.items()
                         if f == fid_a]
            members_b = [r for r, f in self.formation_of.items()
                         if f == fid_b]

            # Block fusion based on attached trolley type
            #   LOW_TROLLEY:   fusion always blocked (no reconfiguration)
            #   HIGH_TROLLEY:  fusion allowed only if merged total ≤ 2
            #   HEAVY_TROLLEY: fusion always allowed
            for rid_check in (rid_a, rid_b):
                att_rid = self._formation_trolley_robot(rid_check)
                if att_rid is not None:
                    uid = self._attached_trolley[att_rid]
                    obs = self.obs_mgr.obstacles.get(uid)
                    if obs is None:
                        continue
                    if obs.kind == ObstacleKind.LOW_TROLLEY:
                        self._set_message(
                            f"Cannot combine while wm{att_rid} has "
                            f"{obs.label} attached. Detach first (T).",
                            HUD_WARN)
                        return
                    if obs.kind == ObstacleKind.HIGH_TROLLEY:
                        merged_n = len(members_a) + len(members_b)
                        if merged_n > 2:
                            self._set_message(
                                f"H.Trolley: max formation size 2 "
                                f"(merged would be {merged_n}). "
                                f"Detach first (T).", HUD_WARN)
                            return
            d = self._closest_pair_distance(members_a, members_b)
            if d > DOCKING_DISTANCE_M:
                self._set_message(
                    f"Docking rejected: closest pair between formations "
                    f"= {d:.2f} m, limit = {DOCKING_DISTANCE_M:.2f} m.",
                    HUD_WARN,
                )
                return

            # Trigger formation host drives the motion
            trigger_host = min(members_a)

            # dock_target = the specific robot the user pointed at (rid_b),
            # NOT the target formation's host.  Robot 5 pressed SHIFT+4
            # → dock adjacent to wm4, even though wm4's host is wm3.
            dock_target = rid_b

            self.docking = {
                "trigger_fid":     fid_a,
                "target_fid":      fid_b,
                "trigger_host":    trigger_host,
                "dock_target":     dock_target,     # approach THIS robot
                "trigger_members": sorted(members_a),
                "target_members":  sorted(members_b),
                "triggered_by":    rid_a,
                "dock_ticks":      0,               # timeout counter
            }
            self._set_message(
                f"Docking: wm{trigger_host}'s formation -> wm{dock_target} "
                f"(triggered from wm{rid_a}, closest pair = {d:.2f} m). "
                f"Moving & aligning...",
                HUD_OK,
            )

        def _try_fission(self, rid: int) -> None:
            if self.docking is not None or self._trolley_docking is not None:
                self._set_message(
                    "Cannot fission while docking/attaching in progress.",
                    HUD_WARN,
                )
                return
            # Block fission when LOW_TROLLEY is attached (no reconfiguration)
            # HIGH_TROLLEY and HEAVY_TROLLEY allow fission
            att_rid = self._formation_trolley_robot(rid)
            if att_rid is not None:
                uid = self._attached_trolley[att_rid]
                obs = self.obs_mgr.obstacles.get(uid)
                if obs and obs.kind == ObstacleKind.LOW_TROLLEY:
                    self._set_message(
                        f"Cannot split while {obs.label} is attached. "
                        f"Detach first (T).", HUD_WARN)
                    return
            members = self._members_of(rid)
            if len(members) <= 1:
                self._set_message(
                    f"wm{rid} is already a split singleton.", HUD_WARN,
                )
                return
            # Configurer FSM handles the kinematic reset (rT_b=I, n=1)
            for m in members:
                self.bus.configurers[m].ingest_rcfg([0, 0, -1])
            # Sim-level: each member becomes its own formation
            for m in members:
                self.formation_of[m] = m

            # Update pinned flag: after fission, each singleton with an
            # attached trolley may no longer meet weight_class → pin it
            for m in members:
                if m in self._attached_trolley:
                    t_uid = self._attached_trolley[m]
                    t_obs = self.obs_mgr.obstacles.get(t_uid)
                    if t_obs is not None:
                        if 1 < t_obs.weight_class:  # singleton = 1
                            t_obs.pinned = True

            self._set_message(
                f"Fission: formation split.  Members {sorted(members)} "
                f"each back to SS.",
                HUD_OK,
            )

        # ----- cmd_vel generation + routing ------------------------------
        def _compute_cmd_from_keys(self) -> Twist:
            s = self.input_state
            if s.q and s.e:
                self.vel_scale = 1.0
                return Twist()
            if s.q and not s.e:
                self.vel_scale = min(VEL_SCALE_MAX,
                                     self.vel_scale + VEL_SCALE_STEP)
            elif s.e and not s.q:
                self.vel_scale = max(VEL_SCALE_MIN,
                                     self.vel_scale - VEL_SCALE_STEP)

            scale = self.vel_scale
            vx = vy = wz = 0.0

            # WASD = differential drive (forward / reverse / rotate)
            # Forward / reverse: W or ↑
            if s.w or s.up:   vx += BASE_LIN_SPEED * scale
            if s.s or s.down: vx -= BASE_LIN_SPEED * scale
            # Yaw (rotate): A (left, +ωz), D (right, −ωz)
            if s.a: wz += BASE_ANG_SPEED * scale
            if s.d: wz -= BASE_ANG_SPEED * scale

            # Arrow keys = holonomic drive (lateral strafe)
            # Strafe: ← (left, +Y), → (right, −Y)
            if s.left:  vy += BASE_LIN_SPEED * scale
            if s.right: vy -= BASE_LIN_SPEED * scale

            return Twist(linear_x=vx, linear_y=vy, angular_z=wz)

        def _distribute_cmd(self, cmd: Twist) -> Dict[int, Twist]:
            """
            Route cmd_vel to formation members.
              * docking active          -> overridden by _docking_cmd()
              * SS selected             -> selected is its own host, route to it
              * FS selected == host     -> route to every formation member
              * FS selected != host     -> ignore (only host accepts teleop)

            When rot_centre_mode == "centroid", the angular component is
            pre-compensated so that the Configurer's Eq 4 transform (which
            assumes rotation about the host) instead produces rotation
            about the formation centroid.  The compensation adds a body-
            frame translational velocity = omega x (H - C) to every
            member's cmd_vel.
            """
            dist: Dict[int, Twist] = {
                rid: Twist() for rid in self.bus.configurers
            }
            if self.docking is not None:
                return dist      # docking controller takes over

            sel = self.selected_id
            if sel not in self.bus.configurers:
                return dist

            members = self._members_of(sel)
            host    = min(members)
            if sel != host and len(members) > 1:
                # FS & selected is a non-host member -> teleop inert
                return dist

            # Possibly compensate cmd for centroid rotation
            routed_cmd = cmd
            if (self.rot_centre_mode == "centroid"
                    and len(members) > 1
                    and abs(cmd.angular_z) > 1e-9):
                # Compute centroid offset in HOST BODY FRAME from the
                # frozen rT_b matrices.  This is a CONSTANT for a given
                # formation (immune to Euler integration drift).
                #
                # C_body = centroid offset from host in host body frame.
                # Compensation = omega x (H - C) in body frame
                #              = omega x (-C_body)
                #              = ( omega * C_body_y,
                #                 -omega * C_body_x )
                cb_x, cb_y = self._centroid_offset_body(sel)
                w = cmd.angular_z
                routed_cmd = Twist(
                    linear_x  = cmd.linear_x  + w * cb_y,
                    linear_y  = cmd.linear_y  - w * cb_x,
                    angular_z = cmd.angular_z,
                )

            for m in members:
                dist[m] = routed_cmd
            return dist

        # ----- docking controller ----------------------------------------
        def _docking_cmd(self) -> Dict[int, Twist]:
            """
            One-tick controller that drives the triggering formation's host
            toward the dock point relative to the **dock_target** robot
            (the specific robot the user pressed SHIFT+N for), rotating to
            align yaw.

            Distinction:
              * dock_target  = the physical robot we approach (e.g. wm4)
              * formation host = min(all members) computed at completion

            Returns the per-robot cmd_vel dict.  Calls _complete_dock()
            when within tolerance, or aborts on timeout.
            """
            dist: Dict[int, Twist] = {
                rid: Twist() for rid in self.bus.configurers
            }
            dk = self.docking
            if dk is None:
                return dist

            # Timeout check (Fix 1)
            dk["dock_ticks"] += 1
            if dk["dock_ticks"] > DOCK_TIMEOUT_TICKS:
                self._set_message(
                    f"Docking timed out after {DOCK_TIMEOUT_TICKS} ticks.  "
                    f"Aborting.",
                    HUD_WARN,
                )
                self.docking = None
                self._last_cmd = Twist()
                return dist

            trig_host   = dk["trigger_host"]
            dock_target = dk["dock_target"]
            if (trig_host not in self.bus.poses
                    or dock_target not in self.bus.poses):
                self.docking = None
                return dist

            trig_pose = self.bus.poses[trig_host]
            targ_pose = self.bus.poses[dock_target]

            # Vector from dock_target to triggering host (world frame)
            dx = trig_pose.x - targ_pose.x
            dy = trig_pose.y - targ_pose.y
            dist_centres = math.hypot(dx, dy)
            if dist_centres < 1e-6:
                dx, dy, dist_centres = 1.0, 0.0, 1.0
            ux, uy = dx / dist_centres, dy / dist_centres

            # Desired dock position = dock_target + u * DOCKED_DISTANCE_M
            desired_x = targ_pose.x + ux * DOCKED_DISTANCE_M
            desired_y = targ_pose.y + uy * DOCKED_DISTANCE_M

            # World-frame position error
            ex_world = desired_x - trig_pose.x
            ey_world = desired_y - trig_pose.y
            pos_err  = math.hypot(ex_world, ey_world)

            # Yaw alignment to dock_target (normalised to (-pi, pi])
            raw_err  = targ_pose.yaw - trig_pose.yaw
            yaw_err  = math.atan2(math.sin(raw_err), math.cos(raw_err))

            # Completion check
            if pos_err < DOCK_POS_TOL_M and abs(yaw_err) < DOCK_YAW_TOL_RAD:
                self._complete_dock()
                return dist

            # Convert world-frame error to trigger host's body frame
            c, s = math.cos(trig_pose.yaw), math.sin(trig_pose.yaw)
            vx_body =  c * ex_world + s * ey_world
            vy_body = -s * ex_world + c * ey_world

            # P-controllers + saturation
            def _clip(v, lo, hi):
                return max(lo, min(hi, v))

            vx = _clip(DOCK_LIN_GAIN * vx_body, -DOCK_LIN_MAX, DOCK_LIN_MAX)
            vy = _clip(DOCK_LIN_GAIN * vy_body, -DOCK_LIN_MAX, DOCK_LIN_MAX)
            wz = _clip(DOCK_ANG_GAIN * yaw_err, -DOCK_ANG_MAX, DOCK_ANG_MAX)

            cmd = Twist(linear_x=vx, linear_y=vy, angular_z=wz)
            # Only the triggering formation moves; target stays still.
            for m in dk["trigger_members"]:
                dist[m] = cmd
            self._last_cmd = cmd    # show in HUD
            return dist

        def _complete_dock(self) -> None:
            """
            Finalise docking: snap trigger host to the exact dock pose
            adjacent to the **dock_target** robot (the specific robot the
            user pressed SHIFT+N for).

            After the snap, rebuild rT_b / host_id / n / FSM state /
            formation_of for every member of the now-unified formation.
            The new host = min(all members) -- this is the naming
            convention and is separate from the physical dock target.
            """
            dk = self.docking
            if dk is None:
                return
            trig_host   = dk["trigger_host"]
            dock_target = dk["dock_target"]
            trig_pose   = self.bus.poses[trig_host]
            targ_pose   = self.bus.poses[dock_target]

            # Snap point: exactly DOCKED_DISTANCE_M from dock_target
            # along the current approach direction.
            dx = trig_pose.x - targ_pose.x
            dy = trig_pose.y - targ_pose.y
            d  = math.hypot(dx, dy)
            if d < 1e-6:
                ux, uy = 1.0, 0.0
            else:
                ux, uy = dx / d, dy / d
            new_trig_host_pose = Pose(
                x   = targ_pose.x + ux * DOCKED_DISTANCE_M,
                y   = targ_pose.y + uy * DOCKED_DISTANCE_M,
                yaw = targ_pose.yaw,       # align yaw to target
            )
            # Rigid offset applied to every member of the triggering formation
            off_x = new_trig_host_pose.x   - trig_pose.x
            off_y = new_trig_host_pose.y   - trig_pose.y
            off_y_yaw = new_trig_host_pose.yaw - trig_pose.yaw
            for m in dk["trigger_members"]:
                pm = self.bus.poses[m]
                self.bus.poses[m] = Pose(
                    x   = pm.x + off_x,
                    y   = pm.y + off_y,
                    yaw = pm.yaw + off_y_yaw,
                )

            # Unify formation metadata for ALL members of both formations.
            unified   = sorted(dk["trigger_members"] + dk["target_members"])
            new_host  = min(unified)
            total_n   = len(unified)
            for m in unified:
                cfg_m = self.bus.configurers[m]
                pm    = self.bus.poses[m]
                ph    = self.bus.poses[new_host]
                if m == new_host:
                    cfg_m.rT_b = np.identity(3)
                else:
                    # rT_b = (R_own)^T @ (-(p_own - p_host))  -- sample form
                    v_world = np.array([pm.x - ph.x, pm.y - ph.y])
                    cy, sy  = math.cos(pm.yaw), math.sin(pm.yaw)
                    R_o     = np.array([[cy, -sy], [sy, cy]])
                    rT_b    = np.identity(3)
                    rT_b[:2, 2] = R_o.T @ (-v_world)
                    cfg_m.rT_b  = np.round(rT_b, 2)
                cfg_m.host_id   = new_host
                cfg_m.n         = total_n
                cfg_m.fsm_state = FSMState.CONFIG
                cfg_m.rcfg      = [0, 0, 0]
                self.formation_of[m] = new_host

            self._set_message(
                f"Docked.  Formation host=wm{new_host}, members={unified}, "
                f"n={total_n}.",
                HUD_OK,
            )
            self.docking = None
            self._last_cmd = Twist()

            # Cancel navigation goals for non-host members of the
            # unified formation.  Only the new host's goal survives
            # (if it had one).  This prevents conflicting nav commands.
            for m in unified:
                if m != new_host and m in self._nav_goals:
                    self._cancel_nav(m)

            # Note: _attached_trolley is keyed by the specific robot that
            # attached (not host). No transfer needed on fusion — the
            # attachment stays on the original robot.

            # Update pinned flag: if formation now meets weight_class, unpin
            for m in unified:
                if m in self._attached_trolley:
                    t_uid = self._attached_trolley[m]
                    t_obs = self.obs_mgr.obstacles.get(t_uid)
                    if t_obs is not None and t_obs.pinned:
                        if total_n >= t_obs.weight_class:
                            t_obs.pinned = False

            # Initialize centroid anchor for the newly formed formation
            n = len(unified)
            cx = sum(self.bus.poses[m].x for m in unified) / n
            cy = sum(self.bus.poses[m].y for m in unified) / n
            self._centroid_anchor[new_host] = (cx, cy)

        # ----- trolley attachment -----------------------------------------

        def _try_trolley_toggle(self) -> None:
            """T key: attach to nearest trolley, or detach if already attached."""
            sel = self.selected_id
            if sel not in self.bus.configurers:
                return
            host = self._host_of(sel)
            if sel != host:
                self._set_message(
                    f"wm{sel} is not the host. Select wm{host} to "
                    f"attach/detach trolley.", HUD_WARN)
                return

            # If already attached (check any member in formation), detach
            att_rid = self._formation_trolley_robot(sel)
            if att_rid is not None:
                uid = self._attached_trolley[att_rid]
                obs = self.obs_mgr.obstacles.get(uid)
                name = obs.label if obs else "trolley"
                # HEAVY_TROLLEY: can only detach as singleton
                if obs and obs.kind == ObstacleKind.HEAVY_TROLLEY:
                    members = self._members_of(att_rid)
                    if len(members) > 1:
                        self._set_message(
                            f"Hv.Trolley: split to singleton first "
                            f"before detaching wm{att_rid}.", HUD_WARN)
                        return
                self._attached_trolley.pop(att_rid)
                if obs:
                    obs.pinned = False  # trolley becomes freely movable again
                self._set_message(
                    f"wm{att_rid} detached from {name}.", HUD_OK)
                return

            # Can't attach while docking or another trolley attach
            if self.docking is not None or self._trolley_docking is not None:
                self._set_message(
                    "Cannot attach while docking/attaching in progress.",
                    HUD_WARN)
                return

            # Find nearest trolley
            members = self._members_of(host)
            formation_size = len(members)
            p = self.bus.poses[host]
            best_obs = None
            best_dist = float("inf")
            for obs in self.obs_mgr.obstacles.values():
                if not obs.is_trolley:
                    continue
                # Already attached to another robot?
                if obs.uid in self._attached_trolley.values():
                    continue
                d = math.hypot(p.x - obs.x, p.y - obs.y)
                if d < best_dist:
                    best_dist = d
                    best_obs = obs
            if best_obs is None or best_dist > TROLLEY_ATTACH_RANGE_M:
                self._set_message(
                    f"No trolley in range ({TROLLEY_ATTACH_RANGE_M:.1f}m).",
                    HUD_WARN)
                return

            # Attach-size validation per trolley type
            #   LOW_TROLLEY:   any formation size allowed
            #   HIGH_TROLLEY:  n ≤ 2 allowed
            #   HEAVY_TROLLEY: singleton only (n=1)
            if best_obs.kind == ObstacleKind.HIGH_TROLLEY:
                if formation_size > 2:
                    self._set_message(
                        f"H.Trolley: max formation size 2 to attach "
                        f"(current n={formation_size}).", HUD_WARN)
                    return
            elif best_obs.kind == ObstacleKind.HEAVY_TROLLEY:
                if formation_size != 1:
                    self._set_message(
                        f"Hv.Trolley: only a singleton (n=1) can attach. "
                        f"Current n={formation_size}.", HUD_WARN)
                    return
            # LOW_TROLLEY: no size restriction

            # Start trolley attachment sequence (similar to docking)
            self._trolley_docking = {
                "host": host,
                "trolley_uid": best_obs.uid,
                "kind": best_obs.kind,
                "ticks": 0,
            }
            self._set_message(
                f"Attaching wm{host} to {best_obs.label} "
                f"(uid={best_obs.uid})...", HUD_OK)

        def _trolley_attach_cmd(self) -> Dict[int, Twist]:
            """Drive the formation toward the trolley attachment pose.
            Returns the per-robot cmd_vel dict.  Calls _complete_trolley_attach()
            when within tolerance, or aborts on timeout."""
            dist: Dict[int, Twist] = {
                rid: Twist() for rid in self.bus.configurers
            }
            td = self._trolley_docking
            if td is None:
                return dist

            td["ticks"] += 1
            if td["ticks"] > DOCK_TIMEOUT_TICKS:
                self._set_message(
                    "Trolley attach timed out. Aborting.", HUD_WARN)
                self._trolley_docking = None
                return dist

            host = td["host"]
            trolley_uid = td["trolley_uid"]
            obs = self.obs_mgr.obstacles.get(trolley_uid)
            if obs is None or host not in self.bus.poses:
                self._trolley_docking = None
                return dist

            h_pose = self.bus.poses[host]
            members = self._members_of(host)

            # Compute target pose depending on trolley type
            if obs.kind == ObstacleKind.LOW_TROLLEY:
                # Edge-to-edge: find nearest edge of trolley to robot
                desired_x, desired_y, desired_yaw = \
                    self._low_trolley_target(h_pose, obs)
            elif (obs.kind == ObstacleKind.HIGH_TROLLEY
                      and len(members) == 2):
                # n=2: drive so formation centroid → trolley centroid,
                # and formation long axis aligns with trolley long axis
                m0, m1 = members[0], members[1]
                p0, p1 = self.bus.poses[m0], self.bus.poses[m1]
                fcx = (p0.x + p1.x) / 2.0
                fcy = (p0.y + p1.y) / 2.0
                # Host target = trolley centre + (host - formation centre)
                desired_x = obs.x + (h_pose.x - fcx)
                desired_y = obs.y + (h_pose.y - fcy)
                # Yaw: align formation long axis with trolley long axis
                fdx = p1.x - p0.x
                fdy = p1.y - p0.y
                formation_yaw = math.atan2(fdy, fdx)
                # Desired yaw correction so formation axis → trolley axis
                yaw_diff = obs.yaw - formation_yaw
                desired_yaw = h_pose.yaw + math.atan2(
                    math.sin(yaw_diff), math.cos(yaw_diff))
            else:
                # HIGH n=1 / HEAVY: centroid-to-centroid alignment
                desired_x = obs.x
                desired_y = obs.y
                desired_yaw = obs.yaw

            # Position error in world frame
            ex_world = desired_x - h_pose.x
            ey_world = desired_y - h_pose.y
            pos_err  = math.hypot(ex_world, ey_world)

            # Yaw error
            raw_err  = desired_yaw - h_pose.yaw
            yaw_err  = math.atan2(math.sin(raw_err), math.cos(raw_err))

            # Completion check
            if pos_err < TROLLEY_DOCK_POS_TOL and abs(yaw_err) < TROLLEY_DOCK_YAW_TOL:
                self._complete_trolley_attach()
                return dist

            # Convert to body frame
            c, s = math.cos(h_pose.yaw), math.sin(h_pose.yaw)
            vx_body =  c * ex_world + s * ey_world
            vy_body = -s * ex_world + c * ey_world

            def _clip(v, lo, hi):
                return max(lo, min(hi, v))

            vx = _clip(DOCK_LIN_GAIN * vx_body, -DOCK_LIN_MAX, DOCK_LIN_MAX)
            vy = _clip(DOCK_LIN_GAIN * vy_body, -DOCK_LIN_MAX, DOCK_LIN_MAX)
            wz = _clip(DOCK_ANG_GAIN * yaw_err, -DOCK_ANG_MAX, DOCK_ANG_MAX)

            cmd = Twist(linear_x=vx, linear_y=vy, angular_z=wz)
            for m in members:
                dist[m] = cmd
            self._last_cmd = cmd
            return dist

        def _low_trolley_target(self, h_pose, obs) -> Tuple[float, float, float]:
            """Compute the target (x, y, yaw) for low-trolley edge-to-edge attach.
            Robot approaches from whichever edge is closest."""
            # Vector from trolley to robot
            dx = h_pose.x - obs.x
            dy = h_pose.y - obs.y

            # Transform to trolley's local frame
            co = math.cos(-obs.yaw)
            so = math.sin(-obs.yaw)
            lx = co * dx - so * dy
            ly = so * dx + co * dy

            hw = obs.half_w
            hh = obs.half_h
            offset = ROBOT_OCCUPANCY_M + TROLLEY_EDGE_OFFSET

            # Find which face is nearest
            faces = [
                ( hw + offset, 0.0, obs.yaw),          # +X face
                (-hw - offset, 0.0, obs.yaw + math.pi),  # -X face
                (0.0,  hh + offset, obs.yaw + math.pi/2),  # +Y face
                (0.0, -hh - offset, obs.yaw - math.pi/2),  # -Y face
            ]
            best = None
            best_d = float("inf")
            for (flx, fly, fyaw) in faces:
                d = math.hypot(lx - flx, ly - fly)
                if d < best_d:
                    best_d = d
                    best = (flx, fly, fyaw)

            # Convert best face position back to world frame
            co_fwd = math.cos(obs.yaw)
            so_fwd = math.sin(obs.yaw)
            wx = obs.x + co_fwd * best[0] - so_fwd * best[1]
            wy = obs.y + so_fwd * best[0] + co_fwd * best[1]
            return (wx, wy, best[2])

        def _complete_trolley_attach(self) -> None:
            """Finalise trolley attachment: snap robot, record attachment."""
            td = self._trolley_docking
            if td is None:
                return
            host = td["host"]
            trolley_uid = td["trolley_uid"]
            obs = self.obs_mgr.obstacles.get(trolley_uid)
            if obs is None:
                self._trolley_docking = None
                return

            h_pose = self.bus.poses[host]
            members = self._members_of(host)

            # Snap to exact target pose
            if obs.kind == ObstacleKind.LOW_TROLLEY:
                tx, ty, tyaw = self._low_trolley_target(h_pose, obs)
            elif (obs.kind == ObstacleKind.HIGH_TROLLEY
                      and len(members) == 2):
                # n=2 attaching: align trolley long axis to formation
                # long axis, then snap formation centroid to trolley centroid
                m0, m1 = members[0], members[1]
                p0, p1 = self.bus.poses[m0], self.bus.poses[m1]
                fdx = p1.x - p0.x
                fdy = p1.y - p0.y
                formation_yaw = math.atan2(fdy, fdx)
                # Rotate trolley to match formation's long axis
                obs.yaw = formation_yaw
                # Target: host snapped so formation centroid = trolley centroid
                fcx = (p0.x + p1.x) / 2.0
                fcy = (p0.y + p1.y) / 2.0
                tx = h_pose.x + (obs.x - fcx)
                ty = h_pose.y + (obs.y - fcy)
                tyaw = formation_yaw
            else:
                tx, ty, tyaw = obs.x, obs.y, obs.yaw

            # Apply snap offset to all formation members
            off_x = tx - h_pose.x
            off_y = ty - h_pose.y
            off_yaw = math.atan2(math.sin(tyaw - h_pose.yaw),
                                  math.cos(tyaw - h_pose.yaw))
            for m in members:
                pm = self.bus.poses[m]
                self.bus.poses[m] = Pose(
                    x=pm.x + off_x, y=pm.y + off_y,
                    yaw=pm.yaw + off_yaw)

            # Record attachment (keyed to the specific robot, not host)
            self._attached_trolley[host] = trolley_uid

            # Set pinned flag based on weight class at attachment time
            members = self._members_of(host)
            if len(members) < obs.weight_class:
                obs.pinned = True
            else:
                obs.pinned = False

            self._trolley_docking = None
            self._last_cmd = Twist()
            if obs.kind == ObstacleKind.HEAVY_TROLLEY:
                self._set_message(
                    f"wm{host}[H] attached to {obs.label}. "
                    f"Fusion allowed. Split to singleton to detach (T).",
                    HUD_OK)
            else:
                self._set_message(
                    f"wm{host} attached to {obs.label}. "
                    f"Combine/split disabled. Press T to detach.",
                    HUD_OK)

        def _is_trolley_attached(self, rid: int) -> bool:
            """True if the robot's formation has an attached trolley."""
            return self._formation_trolley_robot(rid) is not None

        def _formation_trolley_robot(self, rid: int) -> Optional[int]:
            """Return the robot_rid in rid's formation that has a trolley
            attached, or None if no member has one."""
            members = self._members_of(rid)
            for m in members:
                if m in self._attached_trolley:
                    return m
            return None

        def _formation_trolley_obs(self, rid: int) -> Optional[Obstacle]:
            """Return the attached trolley Obstacle for rid's formation,
            or None."""
            tr = self._formation_trolley_robot(rid)
            if tr is None:
                return None
            uid = self._attached_trolley[tr]
            return self.obs_mgr.obstacles.get(uid)

        def _update_attached_trolleys(self) -> None:
            """Move each attached trolley to follow its attached robot.
            Called after bus.tick() so the robot has already moved.

            Pinned-flag enforcement: if obs.pinned is True (set during
            reconfiguration events), the trolley stays put and the
            formation is pinned back to the trolley position instead.
            The pinned flag is toggled by _complete_trolley_attach,
            _complete_dock (fusion), and _try_fission — NOT per-tick.
            """
            to_remove = []
            for att_rid, uid in list(self._attached_trolley.items()):
                obs = self.obs_mgr.obstacles.get(uid)
                if obs is None:
                    to_remove.append(att_rid)
                    continue
                if att_rid not in self.bus.poses:
                    to_remove.append(att_rid)
                    continue

                r_pose = self.bus.poses[att_rid]

                if obs.pinned:
                    # Trolley is immovable (formation too small).
                    # Pin the formation back to the trolley position.
                    members = self._members_of(att_rid)
                    if obs.kind == ObstacleKind.LOW_TROLLEY:
                        # Edge-to-edge: robot pinned at edge offset
                        dx = r_pose.x - obs.x
                        dy = r_pose.y - obs.y
                        d = math.hypot(dx, dy)
                        if d < 1e-6:
                            continue
                        expected_d = (ROBOT_OCCUPANCY_M
                                      + TROLLEY_EDGE_OFFSET + obs.half_w)
                        ux, uy = dx / d, dy / d
                        pin_x = obs.x + ux * expected_d
                        pin_y = obs.y + uy * expected_d
                        shift_x = pin_x - r_pose.x
                        shift_y = pin_y - r_pose.y
                    else:
                        # HIGH / HEAVY: formation centroid → trolley centroid
                        fcx = sum(self.bus.poses[m].x for m in members) / len(members)
                        fcy = sum(self.bus.poses[m].y for m in members) / len(members)
                        shift_x = obs.x - fcx
                        shift_y = obs.y - fcy

                    # Shift entire formation back
                    for m in members:
                        pm = self.bus.poses[m]
                        self.bus.poses[m] = Pose(
                            x=pm.x + shift_x, y=pm.y + shift_y,
                            yaw=pm.yaw)
                    continue

                # Trolley is unpinned — trolley follows robot
                if obs.kind == ObstacleKind.LOW_TROLLEY:
                    dx = r_pose.x - obs.x
                    dy = r_pose.y - obs.y
                    dist_to_r = math.hypot(dx, dy)
                    if dist_to_r < 1e-6:
                        continue
                    expected_d = (ROBOT_OCCUPANCY_M
                                  + TROLLEY_EDGE_OFFSET + obs.half_w)
                    ux, uy = -dx / dist_to_r, -dy / dist_to_r
                    obs.x = r_pose.x + ux * expected_d
                    obs.y = r_pose.y + uy * expected_d
                    obs.yaw = r_pose.yaw
                else:
                    # HIGH / HEAVY: trolley centroid = formation centroid
                    members = self._members_of(att_rid)
                    fcx = sum(self.bus.poses[m].x for m in members) / len(members)
                    fcy = sum(self.bus.poses[m].y for m in members) / len(members)
                    obs.x = fcx
                    obs.y = fcy

                    # HIGH_TROLLEY n=2: align trolley long axis with
                    # the formation's long axis (line between 2 robots)
                    if (obs.kind == ObstacleKind.HIGH_TROLLEY
                            and len(members) == 2):
                        m0, m1 = members[0], members[1]
                        p0 = self.bus.poses[m0]
                        p1 = self.bus.poses[m1]
                        dx = p1.x - p0.x
                        dy = p1.y - p0.y
                        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                            obs.yaw = math.atan2(dy, dx)
                        else:
                            obs.yaw = r_pose.yaw
                    else:
                        obs.yaw = r_pose.yaw

            for rid in to_remove:
                self._attached_trolley.pop(rid, None)

        # ----- collision resolution --------------------------------------
        def _resolve_collisions(self) -> None:
            """
            Mass-weighted collision between DIFFERENT formations.

            Rules
            -----
            1. Any two robots from different formations whose centres are
               closer than DOCKED_DISTANCE_M (= 2·ROBOT_OCCUPANCY_M)
               are considered colliding and push each other apart.
            2. Same-size formations (n_a == n_b): the overlap is split
               50/50, so both formations block each other — neither can
               push the other.
            3. Unequal formations: the LARGER formation pushes the
               SMALLER one.  All correction goes to the smaller side.
               Effectively, the smaller formation cannot push the larger
               one back.

            During docking, all trigger-formation members are exempt from
            colliding with the dock_target so the formation can approach.
            Other inter-formation collisions are resolved normally.
            """
            if not self.bus.configurers:
                return
            ids    = list(self.bus.configurers.keys())
            fmap   = self.formation_of

            # Dragged formation: skip inter-formation collision for it
            drag_fid: Optional[int] = None
            if self._dragging_robot is not None:
                drag_fid = fmap.get(self._dragging_robot)

            # Docking exemption: any trigger member <-> dock_target
            exempt_target: Optional[int] = None
            exempt_members: Optional[set] = None
            if self.docking is not None:
                exempt_target  = self.docking["dock_target"]
                exempt_members = set(self.docking.get(
                    "trigger_members", [self.docking["trigger_host"]]))

            # Pre-compute formation sizes
            fsize: Dict[int, int] = {}
            for r, f in fmap.items():
                fsize[f] = fsize.get(f, 0) + 1

            # Accumulate per-formation translation corrections
            corr: Dict[int, List[float]] = {
                fid: [0.0, 0.0] for fid in set(fmap.values())
            }
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    ri, rj = ids[i], ids[j]
                    fi, fj = fmap[ri], fmap[rj]
                    if fi == fj:
                        continue   # same formation = rigid body, skip
                    # Exempt dragged formation from collision
                    if drag_fid is not None and (fi == drag_fid or fj == drag_fid):
                        continue
                    # Exempt trigger-formation members from dock_target
                    if exempt_target is not None:
                        if (ri == exempt_target and rj in exempt_members) or \
                           (rj == exempt_target and ri in exempt_members):
                            continue
                    pi = self.bus.poses[ri]
                    pj = self.bus.poses[rj]
                    dx = pi.x - pj.x
                    dy = pi.y - pj.y
                    d  = math.hypot(dx, dy)
                    if d < DOCKED_DISTANCE_M:
                        overlap = DOCKED_DISTANCE_M - d
                        if overlap < COLLISION_EPSILON:
                            continue
                        if d < 1e-6:
                            dx, dy, d = 1.0, 0.0, 1.0
                        ux, uy  = dx / d, dy / d

                        ni, nj = fsize[fi], fsize[fj]
                        if ni == nj:
                            # Equal mass -> 50 / 50, mutual block
                            wi, wj = 0.5, 0.5
                        elif ni > nj:
                            # fi is larger -> all push goes to fj
                            wi, wj = 0.0, 1.0
                        else:
                            # fj is larger -> all push goes to fi
                            wi, wj = 1.0, 0.0

                        corr[fi][0] += ux * overlap * wi
                        corr[fi][1] += uy * overlap * wi
                        corr[fj][0] -= ux * overlap * wj
                        corr[fj][1] -= uy * overlap * wj

            for rid in ids:
                cx, cy = corr[fmap[rid]]
                if cx != 0.0 or cy != 0.0:
                    p = self.bus.poses[rid]
                    self.bus.poses[rid] = Pose(
                        x=p.x + cx, y=p.y + cy, yaw=p.yaw,
                    )

        # ----- robot ↔ obstacle collision ------------------------------------
        def _resolve_obstacle_collisions(self) -> None:
            """
            Formation-aware obstacle collision resolution.

            Every member of a formation is checked against every obstacle.
            The correction is applied to the ENTIRE formation as a rigid
            body (shift all members by the same delta), so geometry
            enforcement doesn't undo the push.

            Behaviour by obstacle category:
              * Immovable (wall, pillar, door, human-obstacle):
                  Formation is pushed entirely out.  Obstacle stays fixed.
                  Contact normal recorded for velocity clamping.
              * Moveable (table, chair, trolley):
                  Overlap split 50-50 between formation and obstacle.

            For caster trolleys (HIGH/HEAVY), only caster circles block.

            Must be called AFTER _enforce_formation_geometry so that
            member positions are exact rigid-body positions.
            """
            self._contact_normals.clear()

            if not self.obs_mgr.obstacles:
                return

            formations = self._formations_by_fid()
            eps = COLLISION_EPSILON

            # Two iterations to resolve cascading pushes (3 caused
            # visible jitter from micro-correction oscillation).
            for _iteration in range(2):
                for fid, members in formations.items():
                    # Skip formations being dragged
                    if self._dragging_robot is not None:
                        dr_fid = self.formation_of.get(
                            self._dragging_robot)
                        if dr_fid == fid:
                            continue

                    # Accumulate formation-level correction
                    fx, fy = 0.0, 0.0

                    for rid in members:
                        p = self.bus.poses[rid]

                        for obs in list(self.obs_mgr.obstacles.values()):
                            # Skip obstacles being dragged
                            if obs.uid == self.obs_mgr.dragging_id:
                                continue
                            # Skip attached trolley owned by this formation
                            if obs.uid in self._attached_trolley.values():
                                att_rid = self._formation_trolley_robot(rid)
                                if (att_rid is not None
                                        and self._attached_trolley.get(att_rid) == obs.uid):
                                    continue

                            result = obs.collision_overlap(
                                p.x + fx, p.y + fy, ROBOT_OCCUPANCY_M)
                            if result is None:
                                continue

                            push_x, push_y, overlap = result

                            # Skip micro-overlaps to prevent jitter
                            if overlap < eps:
                                continue

                            if obs.kind == ObstacleKind.HUMAN:
                                # Robots push humans aside — 100% on human,
                                # robot doesn't move.
                                obs.x -= push_x * overlap
                                obs.y -= push_y * overlap
                            elif obs.is_immovable:
                                # 100% push on formation (walls, pillars, doors)
                                fx += push_x * overlap
                                fy += push_y * overlap
                                self._contact_normals.setdefault(
                                    rid, []).append((push_x, push_y))
                            else:
                                # Moveable: split 50-50
                                half = overlap * 0.5
                                fx += push_x * half
                                fy += push_y * half
                                obs.x -= push_x * half
                                obs.y -= push_y * half

                    # Apply accumulated correction to all formation members
                    if fx != 0.0 or fy != 0.0:
                        for rid in members:
                            p = self.bus.poses[rid]
                            self.bus.poses[rid] = Pose(
                                x=p.x + fx, y=p.y + fy, yaw=p.yaw)

        # ----- human ↔ obstacle + human ↔ robot collision -------------------
        def _resolve_human_collisions(self) -> None:
            """
            Push humans out of all obstacles and away from robots.
            Humans cannot walk through anything.  Robots push humans
            aside (handled in _resolve_obstacle_collisions), and
            humans also get pushed out of robots from their own side.
            The human is modelled as a circle with radius = half_w.
            """
            human_obs = [o for o in self.obs_mgr.obstacles.values()
                         if o.kind == ObstacleKind.HUMAN]
            if not human_obs:
                return

            for human in human_obs:
                hr = human.half_w  # human radius

                # Human vs all other obstacles
                for obs in self.obs_mgr.obstacles.values():
                    if obs.uid == human.uid:
                        continue
                    if obs.uid == self.obs_mgr.dragging_id:
                        continue
                    # Human treats everything as a solid wall (pushed out 100%)
                    result = obs.collision_overlap(human.x, human.y, hr)
                    if result is not None:
                        push_x, push_y, overlap = result
                        human.x += push_x * overlap
                        human.y += push_y * overlap

                # Human vs robots (circle-circle)
                # Both are solid to each other — human pushed out 100%.
                # (Robots are already pushed out of humans via obstacle
                #  collision since humans are in _IMMOVABLE.)
                for rid in self.bus.configurers:
                    p = self.bus.poses[rid]
                    dx = human.x - p.x
                    dy = human.y - p.y
                    dist = math.hypot(dx, dy)
                    min_dist = hr + ROBOT_OCCUPANCY_M
                    if dist < min_dist:
                        if dist < 1e-6:
                            dx, dy, dist = 1.0, 0.0, 1.0
                        overlap = min_dist - dist
                        ux, uy = dx / dist, dy / dist
                        # Push human out 100% (robot side handled separately)
                        human.x += ux * overlap
                        human.y += uy * overlap

        def _clamp_vel_against_walls(
                self, distribution: Dict[int, Twist]) -> Dict[int, Twist]:
            """
            For each robot that was in contact with an immovable obstacle
            last frame, remove the velocity component that would push it
            back into the wall.  This prevents teleop-through-walls.

            Works in WORLD frame: converts each robot's body-frame cmd
            to world, subtracts the into-wall component for every active
            contact normal, then converts back to body frame.
            """
            if not self._contact_normals:
                return distribution

            clamped = dict(distribution)
            for rid, normals in self._contact_normals.items():
                cmd = clamped.get(rid)
                if cmd is None:
                    continue
                # Convert body-frame velocity to world frame
                p = self.bus.poses.get(rid)
                if p is None:
                    continue
                ch = math.cos(p.yaw)
                sh = math.sin(p.yaw)
                vx_w = ch * cmd.linear_x - sh * cmd.linear_y
                vy_w = sh * cmd.linear_x + ch * cmd.linear_y

                for nx, ny in normals:
                    # Dot product: how much velocity goes INTO the wall
                    # (against the outward normal)
                    dot = vx_w * (-nx) + vy_w * (-ny)
                    if dot > 0:
                        # Remove the into-wall component
                        vx_w += dot * nx
                        vy_w += dot * ny

                # Convert back to body frame
                vx_b =  ch * vx_w + sh * vy_w
                vy_b = -sh * vx_w + ch * vy_w
                clamped[rid] = Twist(
                    linear_x=vx_b, linear_y=vy_b,
                    angular_z=cmd.angular_z)

                # Also clamp all formation members (they share the cmd)
                members = self._members_of(rid)
                for m in members:
                    if m != rid and m in clamped:
                        mc = clamped[m]
                        mp = self.bus.poses.get(m)
                        if mp is None:
                            continue
                        mch = math.cos(mp.yaw)
                        msh = math.sin(mp.yaw)
                        mvx_w = mch * mc.linear_x - msh * mc.linear_y
                        mvy_w = msh * mc.linear_x + mch * mc.linear_y
                        for nx, ny in normals:
                            dot = mvx_w * (-nx) + mvy_w * (-ny)
                            if dot > 0:
                                mvx_w += dot * nx
                                mvy_w += dot * ny
                        mvx_b =  mch * mvx_w + msh * mvy_w
                        mvy_b = -msh * mvx_w + mch * mvy_w
                        clamped[m] = Twist(
                            linear_x=mvx_b, linear_y=mvy_b,
                            angular_z=mc.angular_z)

            return clamped

        # ----- formation geometry enforcement --------------------------------
        def _enforce_formation_geometry(self) -> None:
            """
            After Euler integration + collision resolution, snap every
            non-host member back to its exact rigid-body position
            relative to the host using the frozen rT_b transform.

            Sign convention of rT_b for non-host member m:
                rT_b_m[:2,2] = R_m.T @ (host_pos - m_pos)
            i.e. vector FROM m TO host, in m's body frame.

            Since all formation members share the same yaw (rigid body),
            R_m == R_host, so to reconstruct member position:
                m_pos = host_pos - R_host @ rT_b_m[:2,2]
            """
            for fid, members in self._formations_by_fid().items():
                if len(members) < 2:
                    continue
                host = min(members)
                hp = self.bus.poses[host]
                ch, sh = math.cos(hp.yaw), math.sin(hp.yaw)
                for m in members:
                    if m == host:
                        continue
                    rTb = self.bus.configurers[m].rT_b
                    tx = rTb[0, 2]   # from-member-to-host, body frame
                    ty = rTb[1, 2]
                    # Rotate into world frame and SUBTRACT to get member pos
                    self.bus.poses[m] = Pose(
                        x   = hp.x - (ch * tx - sh * ty),
                        y   = hp.y - (sh * tx + ch * ty),
                        yaw = hp.yaw,
                    )

        def _pin_centroid(self, distribution) -> None:
            """
            Eliminate Euler orbital drift in centroid-rotation mode.

            After geometry enforcement the formation is a perfect rigid
            body, but the host's Euler-integrated orbital position has a
            small error.  We fix this by tracking the IDEAL centroid
            position and shifting the entire formation to match.

            The ideal centroid moves by only the translational command
            (rotated into world frame by the host yaw).  Any residual
            shift is Euler orbital error and gets corrected.

            Handles ALL multi-robot formations that are currently
            receiving commands (keyboard teleop + all navigating
            formations).

            Args:
                distribution: Dict[int, Twist] mapping host -> raw cmd
                    (PRE centroid-compensation).  Must NOT contain the
                    compensated routed_cmd, or the anchor advance will
                    double-count angular terms and drift.
                    OR a single Twist (legacy compatibility for tests).

            Skipped during docking (formation needs free translation).
            """
            # Legacy: accept a single Twist (tests pass the selected
            # robot's cmd directly).  Convert to a distribution dict.
            if isinstance(distribution, Twist):
                sel = self.selected_id
                if sel is not None and sel in self.bus.configurers:
                    host = self._host_of(sel)
                    distribution = {host: distribution}
                else:
                    distribution = {}

            if self.rot_centre_mode != "centroid" or self.docking is not None:
                self._centroid_anchor.clear()
                return

            # Collect every multi-robot formation that has a nonzero cmd
            formations = self._formations_by_fid()
            for fid, members in formations.items():
                if len(members) < 2:
                    continue
                host = min(members)

                # Determine rotation anchor point.
                # HEAVY_TROLLEY override: use attached robot's position.
                heavy_att = self._formation_trolley_robot(host)
                use_heavy_pin = False
                if heavy_att is not None:
                    uid = self._attached_trolley[heavy_att]
                    obs = self.obs_mgr.obstacles.get(uid)
                    if obs and obs.kind == ObstacleKind.HEAVY_TROLLEY:
                        use_heavy_pin = True

                def _anchor_pos():
                    if use_heavy_pin:
                        pa = self.bus.poses[heavy_att]
                        return pa.x, pa.y
                    n = len(members)
                    return (sum(self.bus.poses[m].x for m in members) / n,
                            sum(self.bus.poses[m].y for m in members) / n)

                # Look up the cmd for this formation's host
                cmd = distribution.get(host, Twist())

                # Skip formations with zero command (not being driven)
                if (abs(cmd.linear_x) < 1e-9
                        and abs(cmd.linear_y) < 1e-9
                        and abs(cmd.angular_z) < 1e-9):
                    # Still need to maintain anchor if it exists
                    if fid in self._centroid_anchor:
                        self._centroid_anchor[fid] = _anchor_pos()
                    continue

                hp = self.bus.poses[host]

                # Compute actual centroid/anchor after geometry enforcement
                ax, ay = _anchor_pos()

                if fid not in self._centroid_anchor:
                    self._centroid_anchor[fid] = (ax, ay)
                    continue

                # Advance anchor by translational command only
                old_cx, old_cy = self._centroid_anchor[fid]
                ch, sh = math.cos(hp.yaw), math.sin(hp.yaw)
                dt = self.bus.dt
                dx_w = (ch * cmd.linear_x - sh * cmd.linear_y) * dt
                dy_w = (sh * cmd.linear_x + ch * cmd.linear_y) * dt
                target_cx = old_cx + dx_w
                target_cy = old_cy + dy_w
                self._centroid_anchor[fid] = (target_cx, target_cy)

                # Shift entire formation to pin centroid
                err_x = target_cx - ax
                err_y = target_cy - ay
                if abs(err_x) > 1e-12 or abs(err_y) > 1e-12:
                    for m in members:
                        p = self.bus.poses[m]
                        self.bus.poses[m] = Pose(
                            x=p.x + err_x, y=p.y + err_y, yaw=p.yaw,
                        )

        def _resync_centroid_anchors(self) -> None:
            """
            After obstacle collision may have shifted formations, update
            centroid anchors to the actual post-collision positions.
            This prevents _pin_centroid from dragging formations back
            through walls on the next frame.
            """
            if not self._centroid_anchor:
                return
            formations = self._formations_by_fid()
            for fid, members in formations.items():
                if fid not in self._centroid_anchor:
                    continue
                if len(members) < 2:
                    continue
                # Recompute anchor from actual positions
                # Check for HEAVY_TROLLEY pin override
                host = min(members)
                heavy_att = self._formation_trolley_robot(host)
                if heavy_att is not None:
                    uid = self._attached_trolley.get(heavy_att)
                    obs = self.obs_mgr.obstacles.get(uid) if uid else None
                    if obs and obs.kind == ObstacleKind.HEAVY_TROLLEY:
                        pa = self.bus.poses[heavy_att]
                        self._centroid_anchor[fid] = (pa.x, pa.y)
                        continue
                n = len(members)
                cx = sum(self.bus.poses[m].x for m in members) / n
                cy = sum(self.bus.poses[m].y for m in members) / n
                self._centroid_anchor[fid] = (cx, cy)

        # ----- rendering -------------------------------------------------
        # ----- navigation / pathfinding -----------------------------------

        def _cancel_nav(self, host: int, msg: Optional[str] = None) -> None:
            """Clean up all nav state for a host."""
            self._nav_goals.pop(host, None)
            self._nav_paths.pop(host, None)
            self._nav_replan_tick.pop(host, None)
            self._nav_fail_count.pop(host, None)
            if msg:
                self._set_message(msg, HUD_WARN)

        def _compute_nav_cmd_for(self, host: int
                                 ) -> Optional[Twist]:
            """
            Compute a body-frame Twist for *one* navigating formation
            (identified by its host).

            Returns None if the host has no active goal, has arrived,
            or no path can be found.  Uses path reuse to avoid
            recomputing every frame, and a fail counter to cancel
            hopelessly stuck goals.
            """
            if host not in self._nav_goals or host not in self.bus.poses:
                return None

            goal = self._nav_goals[host]
            members = self._members_of(host)

            # Use the rotation centre as the "current position"
            rc_x, rc_y = self._rotation_centre(host)
            hp = self.bus.poses[host]

            # Check arrival
            if math.hypot(rc_x - goal[0], rc_y - goal[1]) < PATHFIND_WAYPOINT_TOL:
                self._cancel_nav(host)
                self._set_message(f"wm{host} formation arrived.", HUD_OK)
                return None

            # Decide whether to replan or reuse the existing path.
            # Replan every PATHFIND_REPLAN_INTERVAL ticks, or if no path yet.
            replan_tick = self._nav_replan_tick.get(host, 0)
            need_replan = (host not in self._nav_paths
                           or replan_tick <= 0)

            if need_replan:
                # Build occupancy grid excluding this formation's members
                nav_members = set(members)
                robot_pos = []
                for r in sorted(self.bus.configurers.keys()):
                    if r in nav_members:
                        continue
                    rp = self.bus.poses[r]
                    robot_pos.append((rp.x, rp.y, ROBOT_OCCUPANCY_M))

                grid = self.obs_mgr.build_occupancy_grid(
                    self._world_bounds(),
                    cell_size=PATHFIND_CELL_SIZE,
                    inflate_radius=PATHFIND_INFLATE,
                    robot_positions=robot_pos if robot_pos else None,
                )
                pathfinder = (pathfind_astar if self._pathfind_algo == "astar"
                              else pathfind_dijkstra)
                path = pathfinder(grid, (rc_x, rc_y), goal)

                if path is None or len(path) < 2:
                    # Increment fail counter
                    fails = self._nav_fail_count.get(host, 0) + 1
                    self._nav_fail_count[host] = fails
                    if fails >= PATHFIND_FAIL_LIMIT:
                        self._cancel_nav(
                            host,
                            f"wm{host} nav cancelled (path blocked).")
                        return None
                    # Try using existing path if available
                    path = self._nav_paths.get(host)
                    if path is None or len(path) < 2:
                        return None
                else:
                    # Smooth out grid-aligned waypoints via line-of-sight
                    path = smooth_path(grid, path)
                    # Success — reset fail counter and store path
                    self._nav_fail_count[host] = 0
                    self._nav_paths[host] = path

                self._nav_replan_tick[host] = PATHFIND_REPLAN_INTERVAL
            else:
                self._nav_replan_tick[host] = replan_tick - 1
                path = self._nav_paths.get(host)
                if path is None or len(path) < 2:
                    return None

            # Find next meaningful waypoint (skip those already reached)
            wp_idx = 1
            while wp_idx < len(path) - 1:
                wpx, wpy = path[wp_idx]
                if math.hypot(wpx - rc_x, wpy - rc_y) < PATHFIND_WAYPOINT_TOL:
                    wp_idx += 1
                else:
                    break

            # Look-ahead: instead of aiming at the next waypoint, find
            # a point ~LOOKAHEAD metres along the remaining path.  This
            # rounds off sharp corners and produces much smoother motion.
            LOOKAHEAD = 0.60  # metres ahead on the path
            lx, ly = path[wp_idx]
            remaining = math.hypot(lx - rc_x, ly - rc_y)
            li = wp_idx
            while remaining < LOOKAHEAD and li < len(path) - 1:
                nx, ny = path[li + 1]
                seg = math.hypot(nx - path[li][0], ny - path[li][1])
                remaining += seg
                li += 1
            if li < len(path):
                lx, ly = path[li]

            dx_w = lx - rc_x
            dy_w = ly - rc_y
            dist = math.hypot(dx_w, dy_w)
            if dist < 1e-6:
                return None

            # Speed ramping: decelerate near waypoints / goal to avoid
            # overshoot and the resulting direction snap.
            base_speed = BASE_LIN_SPEED * self.vel_scale
            goal_dist = math.hypot(rc_x - goal[0], rc_y - goal[1])
            # Slow down within 0.5m of goal
            ramp = min(1.0, goal_dist / 0.50)
            speed = max(base_speed * 0.15, base_speed * ramp)
            # Also limit to avoid overshooting the immediate target
            speed = min(speed, dist / DT)

            desired_yaw = math.atan2(dy_w, dx_w)
            ch = math.cos(hp.yaw)
            sh = math.sin(hp.yaw)

            # ── Differential mode ──
            if self._nav_motion == "differential":
                yaw_err = desired_yaw - hp.yaw
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                if abs(yaw_err) > 0.15:
                    wz = max(-BASE_ANG_SPEED * self.vel_scale,
                             min(BASE_ANG_SPEED * self.vel_scale,
                                 yaw_err * 3.0))
                    return Twist(linear_x=0.0, linear_y=0.0, angular_z=wz)
                else:
                    vx_b = min(speed, dist / DT)
                    return Twist(linear_x=vx_b, linear_y=0.0, angular_z=0.0)

            # ── Holonomic mode ──
            elif self._nav_motion == "holonomic":
                vx_w = (dx_w / dist) * speed
                vy_w = (dy_w / dist) * speed
                vx_b =  ch * vx_w + sh * vy_w
                vy_b = -sh * vx_w + ch * vy_w
                return Twist(linear_x=vx_b, linear_y=vy_b, angular_z=0.0)

            # ── Hybrid mode ──
            else:
                vx_w = (dx_w / dist) * speed
                vy_w = (dy_w / dist) * speed
                vx_b =  ch * vx_w + sh * vy_w
                vy_b = -sh * vx_w + ch * vy_w
                yaw_err = desired_yaw - hp.yaw
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                wz = max(-BASE_ANG_SPEED * self.vel_scale,
                         min(BASE_ANG_SPEED * self.vel_scale,
                             yaw_err * 2.0))
                return Twist(linear_x=vx_b, linear_y=vy_b, angular_z=wz)

        def _compute_all_nav_distributions(self
                                            ) -> Tuple[Dict[int, Twist],
                                                       Dict[int, Twist]]:
            """
            Tick navigation for ALL robots that have active goals.

            Returns
            -------
            nav_dist : dict  rid -> Twist
                Per-robot cmd_vel (with centroid compensation applied)
                for every navigating formation member.
            raw_cmds : dict  host -> Twist
                The ORIGINAL (pre-compensation) cmd per host.
                Needed by _pin_centroid to advance the centroid anchor
                without double-counting the compensation terms.

            Fused-formation conflict: if robot A and robot B are in the
            same formation, only the HOST's goal is kept; any non-host
            goals are cancelled.
            """
            nav_dist: Dict[int, Twist] = {}
            raw_cmds: Dict[int, Twist] = {}

            # First pass: normalise goals to hosts and cancel conflicts.
            # If a non-host has a goal, promote to host; if both host
            # and non-host have goals, the host's goal wins.
            goals_to_process: Dict[int, Tuple[float, float]] = {}
            for rid in list(self._nav_goals.keys()):
                if rid not in self.bus.poses:
                    self._cancel_nav(rid)
                    continue
                host = self._host_of(rid)
                if rid != host:
                    # Promote or cancel
                    if host not in self._nav_goals:
                        self._nav_goals[host] = self._nav_goals[rid]
                    self._cancel_nav(rid)
                    rid = host
                goals_to_process[rid] = self._nav_goals[rid]

            # Detect formations with multiple goals (shouldn't happen
            # after normalisation, but guard against it).
            seen_fids: Dict[int, int] = {}   # fid -> host already claimed
            for host in list(goals_to_process.keys()):
                fid = self.formation_of.get(host, host)
                if fid in seen_fids and seen_fids[fid] != host:
                    # Conflict: another host in same formation already
                    # has a goal.  Cancel this one.
                    self._cancel_nav(
                        host,
                        f"wm{host} nav cancelled (formation conflict).")
                    continue
                seen_fids[fid] = host

            # Second pass: compute cmd for each remaining navigating host
            for host in list(self._nav_goals.keys()):
                cmd = self._compute_nav_cmd_for(host)
                if cmd is None:
                    continue

                # Store the raw (pre-compensation) cmd for _pin_centroid
                raw_cmds[host] = cmd

                # Distribute the cmd to all formation members
                # (with centroid compensation, same as _distribute_cmd)
                members = self._members_of(host)
                routed_cmd = cmd
                if (self.rot_centre_mode == "centroid"
                        and len(members) > 1
                        and abs(cmd.angular_z) > 1e-9):
                    cb_x, cb_y = self._centroid_offset_body(host)
                    w = cmd.angular_z
                    routed_cmd = Twist(
                        linear_x  = cmd.linear_x  + w * cb_y,
                        linear_y  = cmd.linear_y  - w * cb_x,
                        angular_z = cmd.angular_z,
                    )
                for m in members:
                    nav_dist[m] = routed_cmd

            return nav_dist, raw_cmds

        # ----- rendering -------------------------------------------------

        def _draw_grid(self) -> None:
            step_m = 1.0
            x = -math.floor(WORLD_W / 2.0)
            while x <= math.ceil(WORLD_W / 2.0):
                p1 = self._world_to_screen(x, -WORLD_H / 2.0)
                p2 = self._world_to_screen(x,  WORLD_H / 2.0)
                pygame.draw.line(self.screen, GRID_COLOUR, p1, p2, 1)
                x += step_m
            y = -math.floor(WORLD_H / 2.0)
            while y <= math.ceil(WORLD_H / 2.0):
                p1 = self._world_to_screen(-WORLD_W / 2.0, y)
                p2 = self._world_to_screen( WORLD_W / 2.0, y)
                pygame.draw.line(self.screen, GRID_COLOUR, p1, p2, 1)
                y += step_m
            ox, oy = self._world_to_screen(0, 0)
            pygame.draw.line(self.screen, (70, 70, 90),
                             (ox, HUD_H), (ox, WINDOW_H), 1)
            pygame.draw.line(self.screen, (70, 70, 90),
                             (0, oy), (WINDOW_W, oy), 1)

        def _draw_hulls(self) -> None:
            HULL_CASTER_ALPHA = 80   # translucent hull over caster trolleys
            for fid, members in self._formations_by_fid().items():
                if len(members) < 2:
                    continue
                xs = [self.bus.poses[m].x for m in members]
                ys = [self.bus.poses[m].y for m in members]
                pad = 0.5
                tl = self._world_to_screen(min(xs) - pad, max(ys) + pad)
                br = self._world_to_screen(max(xs) + pad, min(ys) - pad)
                w = br[0] - tl[0]
                h = br[1] - tl[1]
                rect = pygame.Rect(tl[0], tl[1], w, h)

                # Check if this formation has a caster trolley attached
                has_caster_trolley = False
                for m in members:
                    if m in self._attached_trolley:
                        uid = self._attached_trolley[m]
                        obs = self.obs_mgr.obstacles.get(uid)
                        if obs and obs.kind in _CASTER_TROLLEYS:
                            has_caster_trolley = True
                            break

                if has_caster_trolley:
                    # Draw hull on a translucent surface so trolley
                    # underneath remains visible
                    hull_surf = pygame.Surface((w, h), pygame.SRCALPHA)
                    hull_rect = pygame.Rect(0, 0, w, h)
                    pygame.draw.rect(hull_surf,
                                     HULL_FILL + (HULL_CASTER_ALPHA,),
                                     hull_rect, border_radius=8)
                    pygame.draw.rect(hull_surf,
                                     HULL_EDGE + (HULL_CASTER_ALPHA + 40,),
                                     hull_rect, width=2, border_radius=8)
                    self.screen.blit(hull_surf, (tl[0], tl[1]))
                else:
                    pygame.draw.rect(self.screen, HULL_FILL, rect,
                                     border_radius=8)
                    pygame.draw.rect(self.screen, HULL_EDGE, rect, width=2,
                                     border_radius=8)

                host  = min(members)
                label = f"host=wm{host}  n={len(members)}"
                surf  = self.font_small.render(label, True, HULL_EDGE)
                self.screen.blit(surf, (rect.left + 6, rect.top + 4))

        def _draw_rotation_centres(self) -> None:
            """Draw a crosshair + label at each formation's rotation centre."""
            for fid, members in self._formations_by_fid().items():
                if len(members) < 2:
                    continue
                # Compute the rotation centre for this formation
                rc_x, rc_y = self._rotation_centre(members[0])
                sx, sy = self._world_to_screen(rc_x, rc_y)
                arm = 7  # crosshair arm length in px
                if self.rot_centre_mode == "centroid":
                    col = ROT_CENTRE_COL_C
                    tag = "C"
                else:
                    col = ROT_CENTRE_COL_H
                    tag = "H"
                # Crosshair
                pygame.draw.line(self.screen, col,
                                 (sx - arm, sy), (sx + arm, sy), 2)
                pygame.draw.line(self.screen, col,
                                 (sx, sy - arm), (sx, sy + arm), 2)
                # Small circle
                pygame.draw.circle(self.screen, col, (sx, sy), 4, width=1)
                # Label
                lbl = self.font_small.render(tag, True, col)
                self.screen.blit(lbl, (sx + arm + 2, sy - 8))

        def _robot_under_caster_trolley(self, rid: int) -> bool:
            """True if robot's centre is inside a HIGH/HEAVY trolley body."""
            pose = self.bus.poses[rid]
            for obs in self.obs_mgr.obstacles.values():
                if obs.kind not in _CASTER_TROLLEYS:
                    continue
                if obs.contains_world(pose.x, pose.y):
                    return True
            return False

        def _draw_robot(self, rid: int) -> None:
            cfg  = self.bus.configurers[rid]
            pose = self.bus.poses[rid]
            col  = ROBOT_COLOURS[(rid - 1) % len(ROBOT_COLOURS)]

            cx, cy  = self._world_to_screen(pose.x, pose.y)
            half_px = max(8, self._metres_to_px(ROBOT_SIZE_M))

            # Check if robot is under a caster trolley → draw translucent
            under_trolley = self._robot_under_caster_trolley(rid)
            TRANSLUCENT_ALPHA = 100  # 0=invisible, 255=opaque

            # If under a caster trolley, create a temporary surface
            if under_trolley:
                # Determine bounding box for robot drawing
                extent = max(half_px + 30,
                             self._metres_to_px(DOCKING_DISTANCE_M / 2.0) + 4)
                surf_size = extent * 2
                temp_surf = pygame.Surface((surf_size, surf_size),
                                           pygame.SRCALPHA)
                # Draw offset: centre of temp surface
                ox, oy = extent, extent
            else:
                temp_surf = None
                ox, oy = cx, cy

            target = temp_surf if temp_surf is not None else self.screen

            # 1. Pale blue docking-radius ring (beneath everything)
            dock_radius_px = self._metres_to_px(DOCKING_DISTANCE_M / 2.0)
            ring_col = DOCK_RING
            if self.docking is not None and rid in (
                    self.docking["trigger_host"],
                    self.docking["dock_target"]):
                ring_col = DOCK_ACTIVE_RING
            if dock_radius_px > 2:
                pygame.draw.circle(target, ring_col,
                                   (ox, oy), dock_radius_px, width=1)

            # 2. Occupancy outline (filled subtle disk = collision region)
            occ_px = self._metres_to_px(ROBOT_OCCUPANCY_M)
            pygame.draw.circle(target, (55, 55, 70),
                               (ox, oy), occ_px, width=0)

            # 3. Selection ring
            if rid == self.selected_id:
                pygame.draw.circle(target, SELECTED_RING,
                                   (ox, oy), half_px + 8, width=3)

            # 4. Body (rotated square)
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            corners_local = [
                (-half_px, -half_px),
                ( half_px, -half_px),
                ( half_px,  half_px),
                (-half_px,  half_px),
            ]
            corners_screen = []
            for lx, ly in corners_local:
                # pygame y grows downward -> flip ly
                rx =  c * lx - s * (-ly)
                ry =  s * lx + c * (-ly)
                corners_screen.append((ox + rx, oy - ry))
            pygame.draw.polygon(target, col, corners_screen)
            pygame.draw.polygon(target, (0, 0, 0), corners_screen,
                                width=2)

            # 5. Heading line
            hx = pose.x + (ROBOT_SIZE_M * 1.4) * math.cos(pose.yaw)
            hy = pose.y + (ROBOT_SIZE_M * 1.4) * math.sin(pose.yaw)
            hsx, hsy = self._world_to_screen(hx, hy)
            if temp_surf is not None:
                # Offset heading endpoint relative to temp surface
                hsx_rel = hsx - cx + ox
                hsy_rel = hsy - cy + oy
                pygame.draw.line(target, (255, 255, 255),
                                 (ox, oy), (hsx_rel, hsy_rel), 2)
            else:
                pygame.draw.line(target, (255, 255, 255),
                                 (ox, oy), (hsx, hsy), 2)

            # Blit translucent surface onto main screen
            if temp_surf is not None:
                temp_surf.set_alpha(TRANSLUCENT_ALPHA)
                extent = temp_surf.get_width() // 2
                self.screen.blit(temp_surf,
                                 (cx - extent, cy - extent))

            # 6. Label above body (always opaque for readability)
            host_rid = self._host_of(rid)
            role = "H" if rid == host_rid else "M"
            n    = len(self._members_of(rid))
            lbl  = f"wm{rid}  {role}  n={n}  host=wm{host_rid}"
            surf = self.font_small.render(lbl, True, (240, 240, 240))
            lbl_pos = (cx - surf.get_width() // 2, cy - half_px - 22)
            self.screen.blit(surf, lbl_pos)

        def _draw_hud(self) -> None:
            pygame.draw.rect(self.screen, HUD_BG,
                             (0, 0, WINDOW_W, HUD_H))
            pygame.draw.line(self.screen, (60, 60, 80),
                             (0, HUD_H), (WINDOW_W, HUD_H), 1)

            sel = self.selected_id
            if sel not in self.bus.configurers:
                sel_txt = "Selected: (none)"
            else:
                members = self._members_of(sel)
                host    = min(members)
                if len(members) > 1:
                    role = "HOST" if sel == host else "member (idle)"
                    sel_txt = (
                        f"Selected: wm{sel} [{role}]  "
                        f"formation host=wm{host}, "
                        f"members={sorted(members)}, n={len(members)}"
                    )
                else:
                    sel_txt = (
                        f"Selected: wm{sel}  [split singleton]"
                    )

            cmd         = self._last_cmd
            stop_active = self.input_state.q and self.input_state.e
            scale_col   = HUD_WARN if stop_active else HUD_ACCENT
            if self.docking is not None:
                scale_col = HUD_OK

            line1 = self.font_med.render(sel_txt, True, HUD_ACCENT)
            self.screen.blit(line1, (12, 8))

            status_flag = ""
            if self.docking is not None:
                status_flag = "    [DOCKING]"
            elif stop_active:
                status_flag = "    [E-STOP]"
            rot_tag = ("  rot=CENTROID"
                       if self.rot_centre_mode == "centroid"
                       else "  rot=HOST")
            nav_tag = ""
            n_nav = len(self._nav_goals)
            if n_nav > 0:
                sel = self.selected_id
                sel_host = (self._host_of(sel)
                            if sel in self.bus.configurers else sel)
                if sel_host in self._nav_goals:
                    gx, gy = self._nav_goals[sel_host]
                    nav_tag = f"  NAV→({gx:.1f},{gy:.1f})"
                if n_nav > 1:
                    nav_tag += f"  [{n_nav} robots navigating]"
                elif n_nav == 1 and sel_host not in self._nav_goals:
                    rid = next(iter(self._nav_goals))
                    gx, gy = self._nav_goals[rid]
                    nav_tag = f"  wm{rid} NAV→({gx:.1f},{gy:.1f})"
            scale_line = (
                f"scale={self.vel_scale:.2f}    "
                f"cmd_vel  vx={cmd.linear_x:+.2f}  "
                f"vy={cmd.linear_y:+.2f}  wz={cmd.angular_z:+.2f}"
                + status_flag + rot_tag + nav_tag
            )
            line2 = self.font_med.render(scale_line, True, scale_col)
            self.screen.blit(line2, (12, 32))

            line3 = self.font_small.render(self.last_message, True,
                                           self.last_msg_col)
            self.screen.blit(line3, (12, 58))

            help_left  = ("1-5 select   SHIFT+N dock   SPACE fission   "
                          "Z rot   P algo   M motion   X nav   C clear")
            help_right = ("W/A/S/D diff   I/J/K/L holo   Q↑ E↓ speed   "
                          "Click: spawn/goal   Dbl-click: despawn")
            s1 = self.font_small.render(help_left,  True, (160, 160, 180))
            s2 = self.font_small.render(help_right, True, (160, 160, 180))
            self.screen.blit(s1, (12, 86))
            self.screen.blit(s2, (12, 104))

        def _draw_obstacles(self) -> None:
            """Draw non-caster obstacles (tables, walls, LOW_TROLLEY, etc.)."""
            self.obs_mgr.draw_non_caster(
                self.screen,
                self._world_to_screen,
                self._metres_to_px,
                self.font_small,
            )

        def _draw_caster_trolleys(self) -> None:
            """Draw HIGH/HEAVY trolleys on top of hulls and robots."""
            self.obs_mgr.draw_caster_only(
                self.screen,
                self._world_to_screen,
                self._metres_to_px,
                self.font_small,
            )

        def _draw_occupancy_overlay(self) -> None:
            """
            Translucent yellow inflation halos around all obstacles.
            Visible only while SHIFT is held; serves as a quick visual
            check of the pathfinding clearance zone (= obstacle bound +
            ROBOT_OCCUPANCY_M + PATHFIND_INFLATE).
            """
            if not self.input_state.shift:
                return
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            inflate_m = ROBOT_OCCUPANCY_M + PATHFIND_INFLATE
            for obs in self.obs_mgr.obstacles.values():
                x_min, y_min, x_max, y_max = obs.aabb()
                x_min -= inflate_m; y_min -= inflate_m
                x_max += inflate_m; y_max += inflate_m
                sx_min, sy_min = self._world_to_screen(x_min, y_min)
                sx_max, sy_max = self._world_to_screen(x_max, y_max)
                x = int(min(sx_min, sx_max))
                y = int(min(sy_min, sy_max))
                w = int(abs(sx_max - sx_min))
                h = int(abs(sy_max - sy_min))
                if w > 0 and h > 0:
                    pygame.draw.rect(
                        overlay, (255, 230, 50, 70),
                        pygame.Rect(x, y, w, h), border_radius=6,
                    )
            self.screen.blit(overlay, (0, 0))

        def _draw_paths(self) -> None:
            """Draw navigation paths and goal markers."""
            for rid, path in self._nav_paths.items():
                if len(path) < 2:
                    continue
                pts = [self._world_to_screen(wx, wy) for wx, wy in path]
                pygame.draw.lines(self.screen, (255, 255, 80), False, pts, 2)
            for rid, goal in self._nav_goals.items():
                gx, gy = self._world_to_screen(*goal)
                pygame.draw.circle(self.screen, PATH_GOAL_COL,
                                   (gx, gy), 6, width=2)
                # Cross
                pygame.draw.line(self.screen, PATH_GOAL_COL,
                                 (gx - 5, gy - 5), (gx + 5, gy + 5), 2)
                pygame.draw.line(self.screen, PATH_GOAL_COL,
                                 (gx - 5, gy + 5), (gx + 5, gy - 5), 2)

        def _draw_drag_spawn_preview(self) -> None:
            """Draw a semi-transparent preview of the wall/door/pillar being drag-spawned."""
            p = self._drag_spawn_preview
            if p is None:
                return
            from common.obstacles import OBSTACLE_COLOURS
            kind = p["kind"]
            fill, outline = OBSTACLE_COLOURS[kind]
            cx, cy = self._world_to_screen(p["x"], p["y"])
            hw_px = max(4, self._metres_to_px(p["half_w"]))
            hh_px = max(4, self._metres_to_px(p.get("half_h", OBSTACLE_DIMS[kind][1])))

            if kind == ObstacleKind.PILLAR:
                # Draw circle preview for pillar
                radius_px = hw_px
                size = int(radius_px * 2) + 4
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                centre = size // 2
                pygame.draw.circle(surf, fill + (120,), (centre, centre), int(radius_px))
                pygame.draw.circle(surf, outline + (180,), (centre, centre), int(radius_px), width=2)
                self.screen.blit(surf, (int(cx) - centre, int(cy) - centre))
                # Radius label
                lbl = f"R={p['half_w']:.2f}m"
                lbl_surf = self.font_small.render(lbl, True, (255, 255, 200))
                self.screen.blit(lbl_surf, (cx + 10, cy - 20))
            else:
                c, s = math.cos(p["yaw"]), math.sin(p["yaw"])

                corners_local = [(-hw_px, -hh_px), (hw_px, -hh_px),
                                 (hw_px,  hh_px), (-hw_px,  hh_px)]
                corners = []
                for lx, ly in corners_local:
                    rx =  c * lx - s * (-ly)
                    ry =  s * lx + c * (-ly)
                    corners.append((cx + rx, cy - ry))

                # Draw on a translucent surface
                xs = [pt[0] for pt in corners]
                ys = [pt[1] for pt in corners]
                min_x, max_x = int(min(xs)) - 2, int(max(xs)) + 2
                min_y, max_y = int(min(ys)) - 2, int(max(ys)) + 2
                w = max_x - min_x
                h = max_y - min_y
                if w > 0 and h > 0:
                    surf = pygame.Surface((w, h), pygame.SRCALPHA)
                    local_corners = [(px - min_x, py - min_y) for px, py in corners]
                    pygame.draw.polygon(surf, fill + (120,), local_corners)
                    pygame.draw.polygon(surf, outline + (180,), local_corners, width=2)
                    self.screen.blit(surf, (min_x, min_y))

                # Angle label
                deg = math.degrees(p["yaw"])
                lbl = f"{deg:.0f}\u00b0"
                if kind == ObstacleKind.WALL:
                    length = p["half_w"] * 2
                    lbl += f"  L={length:.2f}m"
                lbl_surf = self.font_small.render(lbl, True, (255, 255, 200))
                self.screen.blit(lbl_surf, (cx + 10, cy - 20))

        def _draw_toolbar(self) -> None:
            """Draw the obstacle selection toolbar on the right side."""
            x0 = WINDOW_W - TOOLBAR_W
            pygame.draw.rect(self.screen, TOOLBAR_BG,
                             (x0, HUD_H, TOOLBAR_W, WINDOW_H - HUD_H))
            pygame.draw.line(self.screen, (60, 60, 80),
                             (x0, HUD_H), (x0, WINDOW_H), 1)

            # Header
            hdr = self.font_med.render("Obstacles", True, TOOLBAR_TEXT)
            self.screen.blit(hdr, (x0 + 8, HUD_H + 8))

            # Algorithm + motion mode indicator
            algo_txt = f"[{self._pathfind_algo.upper()}]"
            algo_surf = self.font_small.render(algo_txt, True, (160, 200, 255))
            self.screen.blit(algo_surf, (x0 + 8, HUD_H + 26))
            mode_short = {"differential": "DIFF",
                          "holonomic": "HOLO",
                          "hybrid": "HYBR"}
            mode_txt = f"[{mode_short[self._nav_motion]}]"
            mode_surf = self.font_small.render(mode_txt, True, (200, 180, 255))
            self.screen.blit(mode_surf, (x0 + 60, HUD_H + 26))

            # Buttons for each obstacle kind
            btn_h = 32
            gap = 4
            y = HUD_H + 44
            for kind in self._toolbar_kinds:
                is_sel = (kind == self._toolbar_selected)
                bg = TOOLBAR_BTN_SEL if is_sel else TOOLBAR_BTN
                rect = pygame.Rect(x0 + 6, y, TOOLBAR_W - 12, btn_h)
                pygame.draw.rect(self.screen, bg, rect, border_radius=4)
                pygame.draw.rect(self.screen, (80, 80, 100), rect,
                                 width=1, border_radius=4)

                # Category indicator (colour dot)
                from common.obstacles import _MOVEABLE, _IMMOVABLE, _DYNAMIC
                if kind in _MOVEABLE:
                    dot_col = (80, 200, 120)   # green
                elif kind in _IMMOVABLE:
                    dot_col = (160, 160, 170)  # grey
                else:
                    dot_col = (240, 140, 80)   # orange
                pygame.draw.circle(self.screen, dot_col,
                                   (x0 + 16, y + btn_h // 2), 4)

                lbl = self.font_small.render(OBSTACLE_LABELS[kind],
                                             True, TOOLBAR_TEXT)
                self.screen.blit(lbl, (x0 + 24, y + (btn_h - 14) // 2))
                y += btn_h + gap

            # Legend at bottom
            leg_y = WINDOW_H - 70
            for dot_col, label in [
                ((80, 200, 120), "Moveable"),
                ((160, 160, 170), "Immovable"),
                ((240, 140, 80), "Dynamic"),
            ]:
                pygame.draw.circle(self.screen, dot_col,
                                   (x0 + 14, leg_y + 6), 4)
                s = self.font_small.render(label, True, (160, 160, 180))
                self.screen.blit(s, (x0 + 24, leg_y))
                leg_y += 18

        def _render(self) -> None:
            self.screen.fill(BG_COLOUR)
            self._draw_grid()
            self._draw_hulls()             # lowest layer — below obstacles & robots
            self._draw_obstacles()
            self._draw_occupancy_overlay() # SHIFT-held translucent inflation halos
            self._draw_paths()
            self._draw_rotation_centres()
            self._draw_caster_trolleys()   # HIGH/HEAVY above hulls, below robots
            for rid in sorted(self.bus.configurers.keys()):
                self._draw_robot(rid)      # robots on top (translucent when under)
            self._draw_drag_spawn_preview()
            self._draw_toolbar()
            self._draw_hud()
            pygame.display.flip()

        # ----- main loop -------------------------------------------------
        def run(self) -> None:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        self._on_keydown(event)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:      # left click
                            self._on_mousedown(event)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self._on_mouseup(event)
                    elif event.type == pygame.MOUSEMOTION:
                        self._on_mousemotion(event)

                self._sync_keyboard_state()

                # Build distribution: docking > navigation > keyboard
                # raw_host_cmds: pre-compensation cmds per host,
                # used by _pin_centroid to advance the centroid
                # anchor without double-counting angular terms.
                raw_host_cmds: Dict[int, Twist] = {}

                if self.docking is not None:
                    distribution = self._docking_cmd()
                    cmd = Twist()  # no user cmd during docking
                elif self._trolley_docking is not None:
                    distribution = self._trolley_attach_cmd()
                    cmd = Twist()
                else:
                    # 1) Compute nav cmds for ALL robots with active goals
                    nav_dist, nav_raw = self._compute_all_nav_distributions()

                    # 2) Determine keyboard cmd for the selected robot.
                    #
                    #    Rules:
                    #    - Only the HOST of a formation accepts teleop.
                    #      Selecting a non-host member → keyboard inert
                    #      (handled by _distribute_cmd returning zeros).
                    #    - If the selected formation's host has an active
                    #      nav goal, nav takes priority over keyboard.
                    #      Keyboard is suppressed (cmd = zero).
                    #    - If the selected robot IS the host and presses
                    #      a movement key while nav is active, cancel
                    #      the nav goal so keyboard teleop takes over.
                    #    - Non-host selection NEVER cancels the host's
                    #      nav goal — the formation keeps navigating.
                    sel = self.selected_id
                    sel_host = self._host_of(sel)
                    sel_is_host = (sel == sel_host)

                    if sel_host in nav_dist:
                        if sel_is_host:
                            # Host is selected AND navigating.
                            # Keyboard overrides nav only if a key
                            # is actually pressed.
                            kb_cmd = self._compute_cmd_from_keys()
                            kb_active = (abs(kb_cmd.linear_x) > 1e-9
                                         or abs(kb_cmd.linear_y) > 1e-9
                                         or abs(kb_cmd.angular_z) > 1e-9)
                            if kb_active:
                                # Cancel nav, use keyboard
                                self._cancel_nav(sel_host)
                                for m in self._members_of(sel_host):
                                    nav_dist.pop(m, None)
                                    nav_raw.pop(m, None)
                                self._set_message(
                                    f"wm{sel_host} nav cancelled "
                                    f"(keyboard override).", HUD_WARN)
                                cmd = kb_cmd
                            else:
                                cmd = Twist()  # nav drives, no keyboard
                        else:
                            # Non-host selected; nav continues, kb inert
                            cmd = Twist()
                    else:
                        # No active nav for this formation.
                        # Only the host accepts keyboard commands.
                        if sel_is_host:
                            cmd = self._compute_cmd_from_keys()
                        else:
                            cmd = Twist()  # non-host: inert
                    self._last_cmd = cmd

                    # 3) Keyboard distribution for the selected formation
                    distribution = self._distribute_cmd(cmd)

                    # 4) Build raw_host_cmds: raw cmd per host for
                    #    _pin_centroid.  Only include the keyboard cmd
                    #    when the selected robot IS the host (non-host
                    #    selection is inert — no cmd reaches the host).
                    if sel_is_host:
                        raw_host_cmds[sel_host] = cmd
                    raw_host_cmds.update(nav_raw)

                    # 5) Merge nav distributions on top (nav overrides
                    #    the zero-cmd for robots that are navigating)
                    distribution.update(nav_dist)

                # Clamp velocities against immovable obstacle contacts
                # from last frame (prevents teleop through walls)
                distribution = self._clamp_vel_against_walls(distribution)

                # Step physics
                self.bus.tick(distribute_cmd=distribution)

                # Resolve inter-formation overlaps
                self._resolve_collisions()

                # Snap formation members to exact rigid-body positions
                # (eliminates Euler integration breathing at any speed)
                self._enforce_formation_geometry()

                # Pin centroid to eliminate Euler orbital drift
                # Uses raw (pre-compensation) cmds so the anchor
                # advance doesn't double-count centroid compensation.
                self._pin_centroid(raw_host_cmds)

                # Resolve robot ↔ obstacle overlaps (formation-aware:
                # pushes the entire formation as a rigid body).
                # Must run AFTER pin_centroid so the obstacle push has
                # the final say and doesn't get undone.
                self._resolve_obstacle_collisions()

                # Resync centroid anchors to actual post-collision
                # positions so pin_centroid doesn't drag the formation
                # back through the wall next frame.
                self._resync_centroid_anchors()

                # Move attached trolleys to follow their formations
                self._update_attached_trolleys()

                # Gather entity positions for door proximity check
                entity_positions = []
                for rid in self.bus.poses:
                    p = self.bus.poses[rid]
                    entity_positions.append((p.x, p.y))
                for obs in self.obs_mgr.obstacles.values():
                    if obs.kind == ObstacleKind.HUMAN:
                        entity_positions.append((obs.x, obs.y))
                self.obs_mgr.update_door_proximity(entity_positions)

                # Tick dynamic obstacles (random walk, door animations)
                self.obs_mgr.tick(DT, self._world_bounds())

                # Resolve human ↔ obstacle and human ↔ robot collisions
                self._resolve_human_collisions()

                # Render
                self._render()
                self.clock.tick(FPS)

            pygame.quit()

    # ------------------------------------------------------------------
    #  Launch
    # ------------------------------------------------------------------
    try:
        TeleopSim().run()
    except KeyboardInterrupt:
        try:
            pygame.quit()
        except Exception:
            pass
        return 130
    return 0


# ============================================================================
#  ENTRY POINT
# ============================================================================
#
#   python Configurer.py                      -> interactive pygame teleop
#   python Configurer.py --demo               -> scripted matplotlib demo (n=3)
#   python Configurer.py --demo --robots 5    -> scripted demo with 5 robots
#   python Configurer.py --demo --headless    -> demo, no matplotlib windows
#
# `from Configurer import Configurer, Twist, ...` triggers NO pygame import.
# ============================================================================

if __name__ == "__main__":
    # When this file is run directly (e.g. `python configurer/open_configurer.py`),
    # add the repo root to sys.path so deferred `from common.obstacles import ...`
    # resolves the same way it does when imported via the package.
    import sys
    from pathlib import Path
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import argparse

    parser = argparse.ArgumentParser(
        description="Configurer: Inter-Reconfiguration FSM standalone entry."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the scripted matplotlib demo instead of the pygame teleop.",
    )
    parser.add_argument(
        "--robots", type=int, default=3,
        help="Number of robots for --demo (2..5, default 3).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="With --demo: skip matplotlib visualisation.",
    )
    args = parser.parse_args()

    if args.demo:
        demo(n_robots=args.robots, visualize=not args.headless)
        sys.exit(0)
    else:
        sys.exit(_run_pygame_teleop())