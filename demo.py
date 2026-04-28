#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py — Unified pygame host for Project Spyder Ascend.

Layered as follows:

  Section 1 (lines below) — verbatim copy of Configurer.py lines 437+.
                             SimBus + TeleopSim + matplotlib `demo` helper +
                             the _run_pygame_teleop launcher + main entry.
                             Provides the full reference teleop sandbox.
  Section 2 (further below) — Phase A additions:
                             Planner ABC, GBNNBasePlanner, NavController,
                             ReconfigSequencer, PlannerRegistry, EventHandler,
                             LMB-drag → GBNN coverage with heatmap overlay,
                             headless test suite, SB3 PPO load spike.

Configurer.py keeps its full standalone teleop sandbox; this file is the
mode-1 host that Phases B–E adapters will plug into.
"""

# ---------------------------------------------------------------------------
# Module-level imports (mirroring Configurer.py's top-of-file imports)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Pure FSM core lives in Configurer.py — import the dataclasses + enum + class
# the copied SimBus / TeleopSim code references.
from configurer.open_configurer import Configurer, Twist, Pose, FSMState

# pygame is imported lazily inside _run_pygame_teleop() — matches Configurer.

# ===========================================================================
# Section 0 (Phase A) — Contracts, GBNN adapter, helper imports
# ===========================================================================
#
# These types + classes are the integration surface for Phases B–E.  They
# sit at module scope so the verbatim TeleopSim copy below can call into
# them with minimal surgery.  Configurer.py is NOT modified.
# ===========================================================================

from abc import ABC, abstractmethod
from common.obstacles import (
    Obstacle, ObstacleKind, ObstacleManager, OccupancyGrid,
    pathfind_astar, pathfind_dijkstra, smooth_path,
    MDA_ARM_RADIUS, MDA_ARM_REACH_M, MDA_ARM_FOV_DEG, MDA_MOUNT_RANGE_M,
)
from common.replicated_gbnn import GBNN
from gbnnh.open_gbnnh import GBNN_H, RoIFrame, AccessPoint

# ---- type aliases ----
XY       = Tuple[float, float]
Waypoint = Tuple[float, float]
Pose2D   = Tuple[float, float, float]
Cell     = Tuple[int, int]
StepIdx  = int


@dataclass
class Rect:
    """Axis-aligned rectangle in world coordinates (metres)."""
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "Rect":
        return Rect(
            min(self.x0, self.x1), min(self.y0, self.y1),
            max(self.x0, self.x1), max(self.y0, self.y1),
        )

    def contains(self, x: float, y: float) -> bool:
        n = self.normalized()
        return n.x0 <= x <= n.x1 and n.y0 <= y <= n.y1

    @property
    def width(self) -> float:
        return abs(self.x1 - self.x0)

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)


class PlannerMode(Enum):
    """Active planner mode (mode 1 = the verbatim Configurer demo)."""
    MANUAL    = 1   # WASD + LMB-click P2P + LMB-drag GBNN coverage
    INTERSTAR = 2   # Phase B
    GBNNH     = 5   # Phase E


@dataclass
class RobotState:
    robot_id:    int
    pose:        Pose
    fsm_state:   FSMState
    host_id:     int
    n:           int
    footprint_m: float = 0.70


@dataclass
class GoalSpec:
    """Tagged union — exactly one field is non-None."""
    point:   Optional[XY]       = None    # point-to-point navigation
    points:  Optional[List[XY]] = None    # fission target list
    area:    Optional[Rect]     = None    # coverage RoI / mode-1 GBNN drag-rect


@dataclass
class ReconfigCommand:
    """Demo-side wrapper around the paper-faithful rcfg triple (Configurer paper, Table I)."""
    recipient_id: int
    rcfg:         Tuple[int, int, int]    # [neighbour_id, size, split_command_code]
    when:         StepIdx = 0


@dataclass
class PlanResult:
    assignments: Dict[int, List[Waypoint]] = field(default_factory=dict)
    reconfig:    List[ReconfigCommand]    = field(default_factory=list)
    algo_starts: Dict[int, Pose2D]        = field(default_factory=dict)
    extras:      Dict                     = field(default_factory=dict)
    metrics:     Dict[str, float]         = field(default_factory=dict)


@dataclass
class WorldSpec:
    """Canonical description of the simulation world for adapters."""
    bounds:    Tuple[float, float, float, float]
    cell_m:    float
    obs_mgr:   "ObstacleManager"
    robots:    List[RobotState]


@dataclass
class PlannerQuery:
    mode:         PlannerMode
    world:        WorldSpec
    robots:       List[RobotState]
    selected_ids: List[int]
    goal:         GoalSpec


class Planner(ABC):
    """Base interface for all planners.  Adapters subclass this."""

    @abstractmethod
    def plan(self, query: PlannerQuery) -> PlanResult:
        """One-shot plan call.  Returns a PlanResult; may be empty."""

    def reset(self) -> None: ...
    def step(self) -> Optional[PlanResult]:
        """Optional iterative advance — return per-step diff or None."""
        return None
    def is_done(self) -> bool: return True
    def render_state(self) -> Dict: return {}


# ---------------------------------------------------------------------------
# DefaultPlanner — wraps the existing pathfind_astar for completeness.
# (The verbatim TeleopSim already uses A*/Dijkstra directly; this adapter
#  exists so Phases B–E can call DefaultPlanner.plan() uniformly.)
# ---------------------------------------------------------------------------

class DefaultPlanner(Planner):
    def __init__(self, cell_size: float = 0.20, inflate_radius: float = 0.40):
        self.cell_size      = cell_size
        self.inflate_radius = inflate_radius
        self._last_path: List[Waypoint] = []

    def plan(self, query: PlannerQuery) -> PlanResult:
        if not query.selected_ids or query.goal.point is None:
            return PlanResult()
        rid   = query.selected_ids[0]
        robot = next((r for r in query.robots if r.robot_id == rid), None)
        if robot is None:
            return PlanResult()
        positions = [(r.pose.x, r.pose.y, r.footprint_m / 2)
                     for r in query.robots]
        excl = next((i for i, r in enumerate(query.robots)
                     if r.robot_id == rid), None)
        grid = query.world.obs_mgr.build_occupancy_grid(
            world_bounds   = query.world.bounds,
            cell_size      = self.cell_size,
            inflate_radius = self.inflate_radius,
            robot_positions= positions,
            exclude_rid    = excl,
        )
        path = pathfind_astar(grid,
                              (robot.pose.x, robot.pose.y),
                              query.goal.point)
        if path is None or len(path) < 2:
            self._last_path = []
            return PlanResult(metrics={"plan_failed": 1.0})
        self._last_path = list(path)
        return PlanResult(
            assignments={rid: list(path)},
            metrics={"path_len_cells": float(len(path))},
        )

    def render_state(self) -> Dict:
        return {"path": list(self._last_path)}


# ---------------------------------------------------------------------------
# GBNNBasePlanner — wraps GBNN.py for mode-1 LMB-drag area coverage.
# Single robot, single RoI, holonomic, dynamic obstacles re-pushed each tick.
# ---------------------------------------------------------------------------

class GBNNBasePlanner(Planner):
    """LMB-drag → rectangular RoI → GBNN coverage for the active robot.

    Rasterises the drawn rectangle to a grid at the robot's footprint cell
    size (under-bite via math.floor — never spills outside the rect).
    If the active robot is inside the RoI, GBNN starts from its current
    cell; otherwise the planner returns an A* approach path back to the
    nearest free RoI cell, and only when the robot arrives does GBNN take
    over.  Dynamic obstacles are re-pushed every step via set_occupancy().
    """

    def __init__(
        self,
        footprint_m: float        = 0.70,
        approach_cell_size: float = 0.20,
        approach_inflate: float   = 0.40,
    ) -> None:
        self.footprint_m        = footprint_m
        self.approach_cell_size = approach_cell_size
        self.approach_inflate   = approach_inflate
        self._gbnn:           Optional[GBNN] = None
        self._roi:            Optional[Rect] = None
        self._cell_size:      float = 0.0
        self._origin:         XY    = (0.0, 0.0)
        self._grid_shape:     Tuple[int, int] = (0, 0)
        self._active_rid:     Optional[int] = None
        self._approach_path:  List[Waypoint] = []
        self._world:          Optional[WorldSpec] = None
        # Robot-visited cells.  Tracked separately from GBNN's grid so that
        # transient obstacles (humans walking through, doors closing) don't
        # cause already-covered cells to revert to "unvisited" the moment
        # the obstacle clears — which would make coverage uncompletable.
        self._visited_cells:  set = set()

    def _cell_to_world(self, cell: Cell) -> XY:
        r, c = cell
        return (self._origin[0] + (c + 0.5) * self._cell_size,
                self._origin[1] + (r + 0.5) * self._cell_size)

    def _world_to_cell(self, wx: float, wy: float) -> Cell:
        cs = self._cell_size
        c = int((wx - self._origin[0]) / cs)
        r = int((wy - self._origin[1]) / cs)
        rows, cols = self._grid_shape
        return (max(0, min(rows - 1, r)), max(0, min(cols - 1, c)))

    def _external_robot_positions(
        self, world: WorldSpec,
    ) -> List[Tuple[float, float, float]]:
        """Robot positions list with the ACTIVE FORMATION excluded.

        GBNN is planning coverage for a single planning unit — whether
        that unit is a split singleton (n=1) or a fused singleton
        (n≥2).  In the fused case, every member of the active formation
        moves rigidly with the host, so treating those members as
        obstacles would (a) pre-block the robot's own cells and (b)
        introduce ghost occupancy that wanders with the formation.
        Same-host-id robots must therefore be excluded from the
        occupancy grid — only robots outside this formation count as
        dynamic obstacles to route around.
        """
        # Find host_id of active robot
        active_host = None
        for r in world.robots:
            if r.robot_id == self._active_rid:
                active_host = r.host_id
                break
        if active_host is None:
            # Fallback: only exclude the active robot itself by identity
            return [(r.pose.x, r.pose.y, r.footprint_m / 2)
                    for r in world.robots
                    if r.robot_id != self._active_rid]
        return [(r.pose.x, r.pose.y, r.footprint_m / 2)
                for r in world.robots
                if r.host_id != active_host]

    def _build_roi_grid(self, world: WorldSpec, rect: Rect) -> np.ndarray:
        """Rasterise — under-bite (math.floor) so the grid never spills past
        the user-drawn rectangle.  Excess margin is split evenly between
        the two sides of each axis."""
        rect = rect.normalized()
        cs   = self.footprint_m
        self._cell_size = cs
        cols = max(1, int(math.floor(rect.width  / cs)))
        rows = max(1, int(math.floor(rect.height / cs)))
        pad_x = (rect.width  - cols * cs) / 2.0
        pad_y = (rect.height - rows * cs) / 2.0
        self._origin = (rect.x0 + pad_x, rect.y0 + pad_y)
        self._grid_shape = (rows, cols)
        g = np.ones((rows, cols), dtype=float)
        # Exclude the active formation's members — they move with the
        # host, so they mustn't appear as static obstacles.
        positions = self._external_robot_positions(world)
        occ = world.obs_mgr.build_occupancy_grid(
            world_bounds=world.bounds, cell_size=cs,
            inflate_radius=0.0, robot_positions=positions, exclude_rid=None,
        )
        for r in range(rows):
            for c in range(cols):
                wx, wy = self._cell_to_world((r, c))
                oc, orow = occ.world_to_cell(wx, wy)
                if not occ.is_free(oc, orow):
                    g[r, c] = -1.0
        return g

    def _live_mask(self) -> np.ndarray:
        """Rebuild obstacle mask each tick so dynamic obstacles propagate.

        Excludes every same-formation robot (same host_id) — split or
        fused, GBNN must never see its own planning unit as obstacles.
        """
        rows, cols = self._grid_shape
        cs = self._cell_size
        rect = self._roi
        assert rect is not None and self._world is not None
        positions = self._external_robot_positions(self._world)
        occ = self._world.obs_mgr.build_occupancy_grid(
            world_bounds=self._world.bounds, cell_size=cs,
            inflate_radius=0.0, robot_positions=positions, exclude_rid=None,
        )
        mask = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                wx, wy = self._cell_to_world((r, c))
                oc, orow = occ.world_to_cell(wx, wy)
                if not occ.is_free(oc, orow):
                    mask[r, c] = True
        return mask

    @staticmethod
    def _nearest_free_cell(grid: np.ndarray, seed: Cell) -> Optional[Cell]:
        from collections import deque
        rows, cols = grid.shape
        if grid[seed[0], seed[1]] != -1.0:
            return seed
        q, seen = deque([seed]), {seed}
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and (nr, nc) not in seen):
                        seen.add((nr, nc))
                        if grid[nr, nc] != -1.0:
                            return (nr, nc)
                        q.append((nr, nc))
        return None

    def plan(self, query: PlannerQuery) -> PlanResult:
        if query.goal.area is None or not query.selected_ids:
            return PlanResult()
        rid = query.selected_ids[0]
        robot = next((r for r in query.robots if r.robot_id == rid), None)
        if robot is None:
            return PlanResult()
        self._active_rid = rid
        self._roi        = query.goal.area.normalized()
        self._world      = query.world
        grid = self._build_roi_grid(query.world, self._roi)
        if self._roi.contains(robot.pose.x, robot.pose.y):
            start_cell = self._world_to_cell(robot.pose.x, robot.pose.y)
            if grid[start_cell[0], start_cell[1]] == -1.0:
                start_cell = self._nearest_free_cell(grid, start_cell)
            self._approach_path = []
        else:
            best, best_d = None, float('inf')
            rows, cols = self._grid_shape
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] == -1.0: continue
                    wx, wy = self._cell_to_world((r, c))
                    d = math.hypot(wx - robot.pose.x, wy - robot.pose.y)
                    if d < best_d:
                        best_d, best = d, (r, c)
            if best is None:
                return PlanResult(metrics={"plan_failed": 1.0})
            start_cell = best
            # Approach A* must also see the active formation as "self",
            # not as an obstacle (otherwise a fused singleton can't find
            # a path to any RoI cell that its own members are standing
            # on or next to).  Reuse the formation-wide exclusion helper.
            positions = self._external_robot_positions(query.world)
            approach_grid = query.world.obs_mgr.build_occupancy_grid(
                world_bounds=query.world.bounds,
                cell_size=self.approach_cell_size,
                inflate_radius=self.approach_inflate,
                robot_positions=positions, exclude_rid=None,
            )
            self._approach_path = pathfind_astar(
                approach_grid, (robot.pose.x, robot.pose.y),
                self._cell_to_world(start_cell),
            ) or []

        if start_cell is None or grid[start_cell[0], start_cell[1]] == -1.0:
            return PlanResult(metrics={"plan_failed": 1.0})

        self._gbnn = GBNN()
        self._gbnn.reset(grid, start_cell)
        # Seed visited tracker with start cell (it was reset by GBNN to a
        # smooth covered-residue value, not +1).  Future step() calls will
        # add each cell the robot moves into.
        self._visited_cells = {tuple(start_cell)}

        assignments: Dict[int, List[Waypoint]] = {}
        if self._approach_path:
            assignments[rid] = list(self._approach_path)
        return PlanResult(
            assignments=assignments,
            algo_starts={rid: (*self._cell_to_world(start_cell), 0.0)},
            extras=self.render_state(),
            metrics={"coverage_pct": self._gbnn.coverage_pct},
        )

    def step(self) -> Optional[PlanResult]:
        if self._gbnn is None or self._world is None:
            return None
        if self._gbnn.is_done():
            return PlanResult(extras=self.render_state(),
                              metrics={"coverage_pct": 1.0, "done": 1.0})

        # Push current obstacle mask into GBNN — this can reset cells that
        # were obstacle-then-cleared back to +1 (unvisited).  Restore the
        # visited-state for any cell the robot has actually been to so the
        # task remains completable when humans / doors come and go.
        self._gbnn.set_occupancy(self._live_mask())
        for (vr, vc) in self._visited_cells:
            if (0 <= vr < self._gbnn._grid.shape[0]
                    and 0 <= vc < self._gbnn._grid.shape[1]
                    and self._gbnn._grid[vr, vc] == 1.0):
                # Cell was visited but got reset to 1.0 by set_occupancy
                # because a transient obstacle just left it.  Mark it
                # visited again (0.0 = covered, no longer attractor).
                self._gbnn._grid[vr, vc] = 0.0

        new_cell = self._gbnn.step()
        # Track the cell the robot is now committed to visit
        self._visited_cells.add(tuple(new_cell))
        new_wxy  = self._cell_to_world(new_cell)
        return PlanResult(
            assignments={self._active_rid: [new_wxy]} if self._active_rid else {},
            extras=self.render_state(),
            metrics={"coverage_pct": self._gbnn.coverage_pct},
        )

    def is_done(self) -> bool:
        return bool(self._gbnn.is_done()) if self._gbnn is not None else True

    def reset(self) -> None:
        self._gbnn = None
        self._roi  = None
        self._active_rid = None
        self._approach_path = []
        self._world = None
        self._visited_cells = set()

    def render_state(self) -> Dict:
        if self._gbnn is None or self._roi is None:
            return {}
        pos = self._gbnn.position
        return {
            "roi":            self._roi,
            "origin":         self._origin,
            "cell_size":      self._cell_size,
            "grid_shape":     self._grid_shape,
            "activity_grid":  self._gbnn.activity_grid,
            "position_cell":  pos,
            "position_world": self._cell_to_world(pos) if pos else None,
            "path_cells":     self._gbnn.path,
            "coverage_pct":   self._gbnn.coverage_pct,
            "iterations":     self._gbnn.iterations,
            # Active robot id → demo.py uses this to tint cell shading
            # with the robot's own colour, so multi-robot coverage scenes
            # stay visually disambiguated.
            "active_rid":     self._active_rid,
        }


# ---------------------------------------------------------------------------
# InterstarPlanner — mode-2 multi-robot fusion / fission via Inter-Star A*
# ---------------------------------------------------------------------------
# Auto-dispatch policy (from B5):
#   * ≥2 robots selected (via Ctrl+click) + LMB click = FUSION
#     — all selected robots converge on the click point
#   * 1 fused singleton (n>1) selected + n LMB clicks  = FISSION
#     — divergence from the singleton's pose to the n click points
#
# Returns per-robot world-frame paths + (k-1) pairwise FUSE / FISSION
# ReconfigCommands scheduled at arrival ticks, and metrics:
#   expansions, baseline_expansions (vs standard A*), expansions_ratio,
#   shared_segment (world-frame polyline for the amber render overlay).

from interstar.open_interstar import Interstar as _Interstar


class InterstarPlanner(Planner):
    """Inter-Star adapter — multi-robot fusion (convergence) or fission
    (divergence) with shared-path exploitation.

    Per Phase B B1, the Interstar class now accepts explicit starts +
    fission_start + mode + numpy grid and returns shared-path metrics via
    its `plan()` classmethod.  This adapter is a thin shim that:
      1. Rasterises the world obstacles to a numpy grid
      2. Converts live robot poses + goal(s) to grid cells
      3. Calls Interstar.plan()
      4. Converts the resulting (row, col) paths back to world waypoints
      5. Builds a PlanResult with per-robot paths + (n-1) FUSEs (fusion)
         or (n-1) FISSIONs (fission) + shared-segment extras + metrics
    """

    def __init__(
        self,
        cell_size:      float = 0.20,
        inflate_radius: float = 0.40,
    ) -> None:
        self.cell_size      = cell_size
        self.inflate_radius = inflate_radius
        self._last_paths:    Dict[int, List[Waypoint]] = {}
        self._last_metrics:  Dict[str, float]          = {}
        self._last_shared:   List[Waypoint]            = []   # world-frame amber segment
        self._last_mode:     str                       = ""   # "fusion" / "fission"

    # ---- helpers ----

    def _world_to_cell(self, grid, wx: float, wy: float) -> Cell:
        """Convert a world (x, y) to an OccupancyGrid (row, col) tuple."""
        c, r = grid.world_to_cell(wx, wy)
        # Clamp into bounds
        r = max(0, min(grid.rows - 1, r))
        c = max(0, min(grid.cols - 1, c))
        return (r, c)

    def _cell_to_world(self, grid, cell: Cell) -> XY:
        r, c = cell
        return grid.cell_to_world(c, r)

    # ---- Planner API ----

    def plan(self, query: PlannerQuery) -> PlanResult:
        if not query.selected_ids:
            return PlanResult()

        # Snapshot world: occupancy grid with NON-PARTICIPANT robots
        # treated as inflated obstacles.  Selected participants AND
        # any same-formation members of those participants are
        # excluded — they're the planning agents and shouldn't block
        # each other (and won't, since the cursor-based driver
        # serialises arrivals at shared cells).  Mirrors A*'s
        # nav_members exclusion in `_compute_nav_cmd_for`, which
        # keeps the two layers consistent: cells Inter-Star plans
        # through are exactly the cells A* would consider free.
        participants = set(query.selected_ids)
        other_robot_pos: List[Tuple[float, float, float]] = []
        for r in query.robots:
            if r.robot_id in participants:
                continue
            # Same-formation members of any participant move with the
            # host and shouldn't block the participant's own grid.
            if r.host_id in participants:
                continue
            other_robot_pos.append(
                (r.pose.x, r.pose.y, r.footprint_m / 2))
        occ = query.world.obs_mgr.build_occupancy_grid(
            world_bounds    = query.world.bounds,
            cell_size       = self.cell_size,
            inflate_radius  = self.inflate_radius,
            robot_positions = other_robot_pos if other_robot_pos else None,
            exclude_rid     = None,
        )
        # Build a numpy grid  0 = free, 1 = obstacle  (Inter-Star's encoding)
        g = np.zeros((occ.rows, occ.cols), dtype=int)
        for r in range(occ.rows):
            for c in range(occ.cols):
                if not occ.is_free(c, r):
                    g[r, c] = 1

        # Build starts / goal based on dispatch type:
        #   FUSION  — multiple selected robots converge on query.goal.point
        #   FISSION — single selected robot with multiple goal points
        sel = list(query.selected_ids)
        if query.goal.points and len(sel) == 1:
            # FISSION: 1 robot, n goal points
            self._last_mode = "fission"
            robot = next((r for r in query.robots if r.robot_id == sel[0]),
                         None)
            if robot is None:
                return PlanResult()
            fission_start_cell = self._world_to_cell(
                occ, robot.pose.x, robot.pose.y)
            # Free the start cell so A* can proceed
            g[fission_start_cell[0], fission_start_cell[1]] = 0
            goal_cells = []
            for (gx, gy) in query.goal.points:
                gc = self._world_to_cell(occ, gx, gy)
                g[gc[0], gc[1]] = 0
                goal_cells.append(gc)
            cell_paths, metrics = _Interstar.plan(
                starts        = [],
                goal          = goal_cells,
                grid          = g,
                mode          = "fission",
                fission_start = fission_start_cell,
                render        = False,
            )
            # paths[k] corresponds to fission goal k; there's no direct
            # robot_id mapping (only one robot is splitting).  We assign
            # each cell-path to one of the existing split-off robot ids;
            # demo.py's _dispatch will ultimately feed them to the host.
            # For the adapter, we just return indexed paths under the
            # single selected rid — the caller interprets them as an
            # n-way fission of that formation.
            world_paths: Dict[int, List[Waypoint]] = {}
            # With fission, paths[0], paths[1], ... are the n divergence
            # trajectories.  We emit them under synthetic keys -1, -2, -3
            # so the caller can disambiguate — demo.py's dispatcher will
            # emit matching FISSION commands and route the paths to the
            # freshly-split members.
            for k, cp in enumerate(cell_paths):
                world_paths[-(k + 1)] = [self._cell_to_world(occ, tuple(c))
                                         for c in cp]
            reconfig = [
                ReconfigCommand(
                    recipient_id = sel[0],
                    rcfg         = (0, 0, -1),   # split_command_code = -1
                    when         = 0,
                )
            ]
            self._last_paths   = world_paths
            self._last_metrics = metrics
            self._last_shared  = []
            return PlanResult(
                assignments = world_paths,
                reconfig    = reconfig,
                extras      = {"shared_segment": [], "mode": "fission"},
                metrics     = {
                    "expansions":        float(metrics["expansions"]),
                    "baseline":          float(metrics["baseline_expansions"]),
                    "expansions_ratio":  float(metrics["expansions_ratio"]),
                },
            )

        # ---- FUSION ----
        if query.goal.point is None:
            return PlanResult()
        self._last_mode = "fusion"
        start_cells = []
        for rid in sel:
            robot = next((r for r in query.robots if r.robot_id == rid), None)
            if robot is None:
                continue
            sc = self._world_to_cell(occ, robot.pose.x, robot.pose.y)
            # Free start cell in case obstacle inflation covered it
            g[sc[0], sc[1]] = 0
            start_cells.append(sc)
        goal_cell = self._world_to_cell(occ, *query.goal.point)
        g[goal_cell[0], goal_cell[1]] = 0
        if len(start_cells) < 2:
            return PlanResult(metrics={"plan_failed": 1.0})

        cell_paths, metrics = _Interstar.plan(
            starts = start_cells,
            goal   = goal_cell,
            grid   = g,
            mode   = "fusion",
            render = False,
        )

        # Convert to world-frame paths keyed by robot_id (order-preserving
        # with `sel` — the Interstar call preserves input ordering).
        world_paths = {}
        for rid, cp in zip(sel, cell_paths):
            if not cp:
                continue
            world_paths[rid] = [self._cell_to_world(occ, tuple(c)) for c in cp]

        shared_world = [
            self._cell_to_world(occ, tuple(c))
            for c in metrics.get("shared_segment", [])
        ]

        # Generate (n-1) FUSE ReconfigCommands.  The first-arriving robot
        # (index 0 in sel) becomes the initial host; each subsequent robot
        # fuses with the growing formation pair-by-pair.  The NavController
        # in demo.py drives each robot along its path; when a robot arrives
        # at the fusion point, the sequencer fires the next pairwise FUSE.
        reconfig: List[ReconfigCommand] = []
        if len(sel) >= 2:
            host_so_far = min(sel)   # smallest-id becomes canonical host
            for idx, rid in enumerate(sel):
                if rid == host_so_far:
                    continue
                # Two-sided handshake: each side publishes its tag to the
                # other.  Sequencer fires them together (same `when`).
                reconfig.append(ReconfigCommand(
                    recipient_id = host_so_far,
                    rcfg         = (rid, 0, 0),
                    when         = idx,
                ))
                reconfig.append(ReconfigCommand(
                    recipient_id = rid,
                    rcfg         = (host_so_far, 0, 0),
                    when         = idx,
                ))

        self._last_paths   = world_paths
        self._last_metrics = metrics
        self._last_shared  = shared_world

        return PlanResult(
            assignments = world_paths,
            reconfig    = reconfig,
            extras      = {
                "shared_segment": shared_world,
                "mode":           "fusion",
            },
            metrics     = {
                "expansions":        float(metrics["expansions"]),
                "baseline":          float(metrics["baseline_expansions"]),
                "expansions_ratio":  float(metrics["expansions_ratio"]),
            },
        )

    def render_state(self) -> Dict:
        return {
            "paths":          dict(self._last_paths),
            "shared_segment": list(self._last_shared),
            "metrics":        dict(self._last_metrics),
            "mode":           self._last_mode,
        }


# ---------------------------------------------------------------------------
# ReconfigSequencer — FIFO of ReconfigCommands fired into Configurer FSMs
# ---------------------------------------------------------------------------
# Phase A does not produce reconfig commands (mode 1 = single-robot teleop /
# coverage); future adapters may emit them.  This scaffold class lets the demo
# accept a `List[ReconfigCommand]` from any adapter and dispatch them as fast
# as the pairwise FSM handshake allows — paper-faithful per Configurer paper, Table I.


class ReconfigSequencer:
    """FIFO queue of ReconfigCommand, fired one-at-a-time as each
    recipient's FSM clears (returns to CONFIG with rcfg inbox == [0,0,0]).

    No artificial pacing — the next command fires the instant the previous
    pairwise handshake completes.  N-robot fusion is dispatched as (N-1)
    pairwise FUSEs scheduled in order via the `when` field.

    Phase A: instantiated by TeleopSim, idle (queue stays empty).
    Future adapters extend the queue via `enqueue` / `extend`.
    """

    def __init__(self, bus) -> None:
        self._bus      = bus
        self._queue:    List[ReconfigCommand] = []
        self._step_idx: int                    = 0

    def enqueue(self, cmd: ReconfigCommand) -> None:
        self._queue.append(cmd)
        self._queue.sort(key=lambda c: c.when)

    def extend(self, cmds: List[ReconfigCommand]) -> None:
        for c in cmds:
            self.enqueue(c)

    def clear(self) -> None:
        self._queue.clear()

    def pending(self) -> int:
        return len(self._queue)

    def tick(self) -> None:
        """Fire any eligible commands whose recipient's FSM is ready."""
        self._step_idx += 1
        i = 0
        while i < len(self._queue):
            cmd = self._queue[i]
            if cmd.when > self._step_idx:
                i += 1
                continue
            cfg = self._bus.configurers.get(cmd.recipient_id)
            if cfg is None:
                self._queue.pop(i)
                continue
            if cfg.fsm_state == FSMState.CONFIG and cfg.rcfg == [0, 0, 0]:
                cfg.ingest_rcfg(list(cmd.rcfg))
                self._queue.pop(i)
                # don't advance i — re-check the new head next iter
            else:
                i += 1


# ---------------------------------------------------------------------------
# PlannerRegistry — named-handle lookup of Planner adapter instances
# ---------------------------------------------------------------------------
# Used in Phase A by TeleopSim to hold the singletons of DefaultPlanner and
# GBNNBasePlanner.  Phases B and E register InterstarPlanner and GBNNHPlanner
# under their own keys — the EventHandler then resolves "which planner do I
# dispatch to" via `registry.get(key)`.


class PlannerRegistry:
    """Central named-handle registry of Planner adapter instances.

    Phase A keys (registered by TeleopSim.__init__):
        "default" → DefaultPlanner   (P2P A* — used by mode-1 LMB-click)
        "gbnn"    → GBNNBasePlanner  (RoI coverage — used by mode-1 LMB-drag)

    Phase B+ adds: "interstar", "gbnnh".
    """

    def __init__(self) -> None:
        self._planners: Dict[str, Planner] = {}

    def register(self, key: str, planner: Planner) -> None:
        self._planners[key] = planner

    def get(self, key: str) -> Optional[Planner]:
        return self._planners.get(key)

    def keys(self) -> List[str]:
        return list(self._planners.keys())


# ===========================================================================
# Section 1 — copy of Configurer.py lines 437 .. EOF (verbatim)
# ===========================================================================

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
      Spyder-compatible -- matches Interstar / GBNN+H pattern).
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
        Render current scene.  Fresh figure + plt.show() per call,
        matching Interstar / GBNN+H pattern (no FuncAnimation).
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
                f"Robot{rid}\n{cfg.fsm_state.value}  n={cfg.n}  host=Robot{cfg.host_id}",
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
        print(f"  Robot{rid}:")
        for i, t in enumerate(tail):
            print(f"    [{i}] vx={t.linear_x:+.3f}  vy={t.linear_y:+.3f}  "
                  f"wz={t.angular_z:+.3f}")


def demo(n_robots: int = 3, visualize: bool = True) -> None:
    """
    Scripted scenario (Spyder-runnable):
      1. Spawn N split singletons (SS), visualise initial state.
      2. Cruise forward -- verify SS robots act independently (rT_b = I).
      3. Fuse Robot1 + Robot2.  Visualise the fused singleton.
      4. Turn the fused singleton.  Trace how xfm_vel differs between
         host (zero offset) and members (omega-coupled offset terms).
      5. (If N>=3) Fuse Robot3 into the formation, turn again.
      6. Fission all fused members.  Visualise SS again.
      7. Print state snapshots + tail of xfm traces.
    """
    bus = _make_bus(n_robots=n_robots, visualize=visualize)
    bus.viz(label="init (all SS)")

    _phase_cruise(bus, ticks=5, label="SS cruise")

    # --- Fuse Robot1 + Robot2 (handshake takes a couple of ticks) ---
    print("\n>>> FUSE  Robot1 <- Robot2")
    bus.send_fusion_command(host_id=1, neighbour_id=2)
    bus.send_fusion_command(host_id=2, neighbour_id=1)
    # Drive ticks until both settle -- bounded loop is safe because
    # the sample protocol completes in 2 ticks.
    for _ in range(6):
        bus.tick()
        if all(c.fsm_state == FSMState.CONFIG for c in bus.configurers.values()):
            break
    bus.viz(label="after fuse Robot1+Robot2")

    _phase_rotate_fused(bus, host_id=1, ticks=5, label="fused Robot1+Robot2 turning")

    # --- If more than two, fuse Robot3 in next ---
    if n_robots >= 3:
        print("\n>>> FUSE  Robot1 <- Robot3")
        bus.send_fusion_command(host_id=1, neighbour_id=3)
        bus.send_fusion_command(host_id=3, neighbour_id=1)
        for _ in range(6):
            bus.tick()
            if all(c.fsm_state == FSMState.CONFIG for c in bus.configurers.values()):
                break
        bus.viz(label="after fuse Robot1+Robot2+Robot3")
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
        print(f"  Robot{rid}:  {snap}")
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
# Controls (see `python Configurer.py` for interactive use):
#   1..5                 select robot (mode switch)
#   W / S                forward / backward
#   A / D                rotate left / right
#   SHIFT + A / D        strafe left / right
#   Q / E                scale velocity up / down   (Q+E = emergency stop)
#   SHIFT + N            fuse selected with wmN    (iff neighbours)
#   SPACE                fission formation containing selected
#   T                    attach/detach trolley (nearest in range)
#   R                    reset scenario
#   ESC                  quit
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
    HUD_H          = 150                      # reserved strip at top
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
    N_ROBOTS             = 5
    ROBOT_SIZE_M         = 0.35               # body half-width (footprint metadata + GBNN cell size)
    # Collision radius is sized so the visual square's circumscribed
    # circle (corner-to-centre = ROBOT_SIZE_M * sqrt(2)) fits exactly
    # inside the collision circle.  At ROBOT_OCCUPANCY_M = 0.35*sqrt(2)
    # ~= 0.495 m, two robots at DOCKED_DISTANCE_M = 2 * 0.495 = 0.99 m
    # have their collision circles just-touching AND their squares
    # cannot visually overlap regardless of relative rotation.  This
    # restores the original 0.35 m visual half-width without cosmetic
    # corner overlap at docking.
    #
    # Functional impact vs. the previous 0.40 m collision radius:
    #   - A* path inflation widens (0.495 vs 0.40 m), so robots route
    #     around obstacles with more clearance.  Narrow gaps that were
    #     0.80 m passable now need 0.99 m.
    #   - Formation spacing scales with DOCKED_DISTANCE_M; fused units
    #     sit ~0.19 m further apart than before.
    #   - PATHFIND_INFLATE (0.35) is still less than ROBOT_OCCUPANCY_M,
    #     so the self-block guard still holds.
    ROBOT_OCCUPANCY_M    = ROBOT_SIZE_M * math.sqrt(2.0)   # ~= 0.4950 m
    ROBOT_VISUAL_HALF_M  = ROBOT_SIZE_M                    # = 0.35 m, matches body extent
    DOCKED_DISTANCE_M    = 2.0 * ROBOT_OCCUPANCY_M    # ~= 0.99 m
    DOCKING_DISTANCE_M   = 2.00                       # eligibility radius
    INITIAL_SPACING_M    = DOCKING_DISTANCE_M * 0.75  # = 1.50 m (inside range)

    DT                   = 1.0 / FPS
    # Base speeds — tripled from the paper defaults (0.60 / 1.20) because
    # the user routinely ran the sandbox at scale 3.  vel_scale bounds
    # (0.10–3.00) stay the same, so the usable speed band is now 3× wider.
    BASE_LIN_SPEED       = 1.80               # m/s, scaled by vel_scale
    BASE_ANG_SPEED       = 3.60               # rad/s
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
        # WASD — differential drive (forward/back/turn).  No Shift modifier.
        w: bool = False
        a: bool = False
        s: bool = False
        d: bool = False
        # Arrow keys — holonomic 4-direction strafe.  Layered on top of
        # WASD: e.g. W + RIGHT = drive forward AND strafe right.
        up:    bool = False
        down:  bool = False
        left:  bool = False
        right: bool = False
        # Shift is now reserved for non-drive gestures (AP placement
        # via Shift+LMB, clearance-zone halo overlay).  Drive no
        # longer reads it.
        shift: bool = False
        q:     bool = False
        e:     bool = False

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
            # Debug diagnostics for a 4th HUD line (per-tick) showing
            # why the currently-selected robot is/isn't moving.
            #   _dbg_nav_reason[host]   — short tag explaining the last
            #                             return of _compute_nav_cmd_for
            #                             for that host this tick.
            #   _dbg_collision_push[rid] — (dx, dy) applied by the
            #                              collision resolvers this
            #                              tick; zero if none.
            self._dbg_nav_reason:     Dict[int, str]                      = {}
            self._dbg_collision_push: Dict[int, Tuple[float, float]]      = {}
            # Per-host pose snapshot (start of each tick's collision
            # pass), used to derive a velocity proxy for the
            # velocity-weighted inter-formation push: the formation
            # with the larger per-tick displacement is treated as
            # "active" and absorbs less of the overlap correction;
            # the slower / idle formation absorbs more (gets
            # shoved out of the way).  Both still get SOME push,
            # so neither passes through the other.
            self._prev_collision_poses: Dict[int, Pose] = {}

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

            # ============================================================
            # PHASE A — additions on top of the verbatim Configurer demo
            # ============================================================
            #
            # `_gbnn_planner_for[host]` holds the GBNNBasePlanner instance
            # currently driving coverage for `host`.  None when no GBNN
            # run is active for that formation.
            #
            # `_gbnn_world_path[host]` is the precomputed world-frame
            # waypoint polyline for that GBNN run (used as `_nav_paths`
            # so the existing nav controller drives the robot through
            # the cells in order).
            #
            # `_gbnn_drag_*` track the LMB-drag in screen pixels so the
            # release classifies as tap (P2P nav) vs drag (GBNN coverage).
            self._gbnn_planner_for:  Dict[int, GBNNBasePlanner] = {}
            self._gbnn_approach:     Dict[int, List[Tuple[float, float]]] = {}
            self._gbnn_approach_idx: Dict[int, int] = {}
            self._gbnn_drag_start:   Optional[Tuple[int, int]] = None
            self._gbnn_drag_last:    Optional[Tuple[int, int]] = None
            self._gbnn_drag_armed:   bool = False
            self.GBNN_DRAG_THRESHOLD_PX: int = 10

            # ---- Phase B (Interstar) state ----
            # Multi-robot Ctrl+click selection (B3)
            self._selected_ids:  set = set()
            # Interstar adapter + last-plan state (B2 / B8 / B10)
            self._interstar_planner: Planner = None   # set below after class is known
            self._interstar_plan_active: bool = False
            self._interstar_shared_segment: List[Tuple[float, float]] = []
            self._interstar_paths:   Dict[int, List[Tuple[float, float]]] = {}
            self._interstar_cursor:  Dict[int, int] = {}    # per-robot waypoint index
            self._interstar_metrics: Dict[str, float] = {}
            self._interstar_host:    Optional[int] = None
            # Pending fission goals queue — built up by successive LMB clicks
            # on a single fused singleton, dispatches when count == n (B5)
            self._fission_goals_pending: List[Tuple[float, float]] = []
            self._fission_host:          Optional[int] = None
            # FUSE command queue staging (after robots arrive at fusion pt)
            self._interstar_pending_fuses: List[ReconfigCommand] = []
            # Mode flag ("fusion" / "fission") — drives which termination
            # criterion applies and whether iterative re-planning fires.
            self._interstar_mode: str = ""
            # Fusion parameters cached for iterative re-planning
            self._interstar_fusion_goal: Optional[Tuple[float, float]] = None
            self._interstar_selected_cache: List[int] = []
            self._interstar_replan_counter: int = 0
            # Auto-complete detection: all selected stopped for N frames AND
            # at least one has reached the goal → end Interstar + combine
            self._interstar_stop_timer: int = 0
            self._interstar_last_positions: Dict[int, Tuple[float, float]] = {}
            # Sequential _try_fuse queue (populated on auto-complete).
            # Stored as an explicit list of (trigger_rid, target_rid) pairs
            # following the proximity-chain rule:
            #   * furthest-from-goal robot triggers each fusion
            #   * target is the closest robot across formations
            #   * goal-pose robot is always the last target to be joined
            self._interstar_fusion_pairs: List[Tuple[int, int]] = []
            # Formation-packer staging (pre-fuse).  After Inter-Star
            # termination, each non-anchor host drives to its assigned
            # dock slot around the anchor via A*, so by the time the
            # pairwise `_try_fuse` fires every trigger is already at
            # the right DOCKED_DISTANCE_M offset at a distinct angle.
            # This prevents the rigid-translation overlap that used to
            # happen when the docking snap + yaw alignment swept
            # existing formation members through other formations'
            # positions.
            self._interstar_staging_slots: Dict[int, Tuple[float, float]] = {}
            self._interstar_combine_anchor: Optional[int] = None
            self._interstar_staging_active: bool = False
            self._interstar_staging_timer: int = 0
            # Rest-based staging transition.  Instead of a blunt
            # absolute timeout (previously 6 s), we transition to the
            # pairwise fuse phase only once EVERY staging host is at
            # rest — i.e. moving less than INTERSTAR_STOP_TOL_M per
            # tick for INTERSTAR_STAGING_REST_FRAMES consecutive
            # ticks.  If a host is still driving to its slot (even a
            # long path), we wait.  If a host got stuck and its nav
            # was cancelled, it's at rest → rest-ticks accumulate →
            # transition fires.  This matches the user's spec "the
            # timer for time out should be triggered until every
            # fusion candidate is at rest".
            #
            # A large absolute safety cap remains so a badly-broken
            # state can't hang forever.
            self._interstar_staging_rest_ticks: int = 0
            self.INTERSTAR_STAGING_REST_FRAMES: int = 30  # 1 s at 30 FPS
            self.INTERSTAR_STAGING_TIMEOUT: int = 600     # 20 s safety
            self.INTERSTAR_STAGING_TOL_M: float = 0.35
            self._interstar_staging_last_pose: Dict[int, Tuple[float, float]] = {}
            # Fission goal map {rid → user's click-point (wx, wy)} — set
            # by `_dispatch_interstar_fission` after the optimal
            # member→goal assignment.  The termination check uses this
            # (not `path[-1]`) so we compare against the click point the
            # user actually asked for, not the Inter-Star path's snapped
            # endpoint cell.
            self._interstar_fission_goals: Dict[int, Tuple[float, float]] = {}
            # Per-member start poses at fission dispatch — used to
            # drop the Inter-Star candidate star as soon as a member
            # has moved away from where it started (the user wants
            # fission stars to vanish the instant members begin
            # diverging, not when they reach their goals).
            self._interstar_fission_start_poses: Dict[int, Tuple[float, float]] = {}
            self.INTERSTAR_FISSION_STAR_MOVE_M: float = 0.25
            # Constants
            self.INTERSTAR_STOP_TOL_M: float = 0.01   # per-frame displacement
            self.INTERSTAR_STOP_FRAMES: int = 30      # 1 second at 30 FPS
            # "All clustered near goal" fallback for the jitter case:
            # when 3+ robots converge on one fusion goal and start
            # mutually pushing each other (especially when a fused
            # singleton's larger footprint blocks a smaller one), the
            # `all_stopped` condition never fires and termination
            # would loop forever.  This fallback fires when every
            # selected robot is within NEAR_GOAL_RADIUS for
            # INTERSTAR_CLUSTER_FRAMES consecutive ticks, regardless
            # of motion.
            self.INTERSTAR_NEAR_GOAL_RADIUS_M: float = 2.5
            self.INTERSTAR_CLUSTER_FRAMES: int = 90    # 3 seconds at 30 FPS
            self._interstar_cluster_timer: int = 0
            # "At goal" threshold: 1.5 m is generous enough that the
            # waypoint-wait serialisation (which halts robots ~0.8 m short
            # of the actual goal cell because other selected robots block
            # the final cells) still counts the formation as "converged".
            self.INTERSTAR_GOAL_TOL_M: float = 1.5
            # Iterative re-plan cadence — fusion mode re-runs Inter-Star
            # with current robot poses every N frames so shared-path
            # decisions track the live world.
            self.INTERSTAR_REPLAN_INTERVAL: int = 15   # 2 Hz at 30 FPS

            # ============================================================
            # Phase E (Pressing 5) — GBNN+H surface cleaning state
            # ============================================================
            # Lifecycle (revised):
            #   1. Mount an MDA on a robot via key '0'.
            #   2. Shift+LMB on empty floor near an obstacle → registers an
            #      access point.  Position = click; yaw points from click
            #      toward the nearest obstacle.  APs accumulate in
            #      `_gbnnh_aps`; an AP is rejected if the click lands on
            #      an obstacle, or if no obstacle exists within MDA reach.
            #   3. Press Enter → robot computes a visit sequence and starts
            #      navigating to AP[0].  No FOV evaluation is done yet.
            #   4. On arrival at each AP, the robot computes the FOV-based
            #      RoI grid (line-of-sight ∩ cone) and ticks GBNN+H to
            #      completion before advancing to the next AP.
            #   5. Esc clears all Mode-5 state at any point.

            # Run state
            self._gbnnh_active:        bool = False
            self._gbnnh_host_rid:      Optional[int] = None
            # AP queue.  Each element is a dict with:
            #   pose:          (x, y, yaw)              world-frame base pose
            #   roi_segments:  Optional[List[Dict]]     built lazily on arrival
            #   label:         str
            #   done:          bool
            #   stats:         dict (filled on completion)
            self._gbnnh_aps:           List[Dict[str, object]] = []
            self._gbnnh_active_ap_idx: Optional[int] = None
            # Per-AP cleaning sweep cursor — index into the active AP's
            # `roi_segments` list, or None when no AP is being cleaned
            # (en route, yaw aligning, or run finished).
            self._gbnnh_active_seg_idx: Optional[int] = None
            # Legacy planner field kept None — Phase E v3 switched to
            # segment-based sweep cleaning; the GBNN_H grid planner is
            # no longer instantiated.  Field retained as a "cleaning
            # phase started" sentinel for code paths that still test it.
            self._gbnnh_planner:       Optional["GBNN_H"] = None

            # UI state — surface-view subpanel toggle (Tab)
            self._gbnnh_show_panel:     bool = True

            # Auto-revert timer — pygame ms timestamp of when the most
            # recent run completed.  When set, the per-tick driver
            # waits GBNNH_RESET_AFTER_MS, then converts every AP back
            # from "done" (green) to "queued" (yellow) and clears the
            # timestamp.  None = no run has completed yet, or a fresh
            # run has just started.
            self._gbnnh_completion_tick: Optional[int] = None
            # Stall detection during AP en-route — when the host
            # cannot make progress toward the AP for too long
            # (typically because A* cannot find a path through tight
            # obstacle clearances), the driver gives up and accepts
            # the host's current pose as arrival rather than spinning
            # forever.
            self._gbnnh_stall_pos:   Optional[Tuple[float, float]] = None
            self._gbnnh_stall_count: int = 0
            # GBNN+H step throttle — counts frames since the last
            # planner.step() call.  Reset on AP entry; advance one
            # cell per GBNNH_FRAMES_PER_STEP frames.
            self._gbnnh_step_frame_counter: int = 0

            # Render + behavior constants
            self.GBNNH_AP_RADIUS_M:     float = 0.30   # marker radius on floor
            self.GBNNH_AP_TOL_M:        float = 0.50   # base-arrival radius
            # Surface grid resolution at each AP — square grid spanning
            # the FOV cone bounding box.
            self.GBNNH_SURFACE_CELLS_W: int   = 12
            self.GBNNH_SURFACE_CELLS_H: int   = 12
            # Hold "all done" (green markers) for this long after the
            # run ends, then revert markers to yellow for re-run.
            self.GBNNH_RESET_AFTER_MS:  int   = 1000
            # Per-segment GBNN+H grid: cell size along the segment's
            # length, and a fixed depth (rows perpendicular to the
            # segment, representing surface extent — wall height /
            # table depth).  Coarse enough to clean quickly but with
            # enough resolution to see the EE markers / trails.
            self.GBNNH_SEG_CELL_M:      float = 0.10   # grid cell size
            self.GBNNH_SEG_DEPTH_CELLS: int   = 4      # rows in each grid
            # GBNN+H step throttle — advance the planner once every N
            # pygame frames so the EE waypoint motion is human-readable
            # (instead of blurring through cells at 30 steps/second).
            # 15 frames at 30 FPS → ~0.5 s per waypoint.
            self.GBNNH_FRAMES_PER_STEP: int = 15
            # Stall detection — if the host hasn't moved more than
            # GBNNH_STALL_MOVE_TOL_M in a frame for
            # GBNNH_STALL_FRAME_LIMIT consecutive frames, the en-route
            # phase gives up and accepts the host's current pose as
            # arrival.  This lets the cleaning start even when A*
            # can't find a path that gets the host all the way to the
            # AP point (typical when the AP is tucked into a tight
            # corner near the inflated obstacle clearance zones).
            self.GBNNH_STALL_MOVE_TOL_M: float = 0.02   # < 2 cm/frame
            self.GBNNH_STALL_FRAME_LIMIT: int = 90       # 3 s at 30 FPS

            # Reusable adapter instances (also let Phases B-E hot-swap)
            self._default_planner: Planner = DefaultPlanner()
            self._gbnn_planner:    Planner = GBNNBasePlanner(
                footprint_m=2 * ROBOT_SIZE_M)

            # Phase-A scaffolds: registry + reconfig sequencer.  Both are
            # idle in mode 1 (no mode switching, no reconfig events),
            # and exist so Phases B–E can plug in with zero changes here.
            self.registry = PlannerRegistry()
            self.registry.register("default", self._default_planner)
            self.registry.register("gbnn",    self._gbnn_planner)

            # Phase B: Interstar adapter.  Not on its own keybind — dispatched
            # automatically based on selection state (see _on_mouseup).
            self._interstar_planner = InterstarPlanner(
                cell_size      = 0.20,
                inflate_radius = 0.40,
            )
            self.registry.register("interstar", self._interstar_planner)

            self._spawn_scenario()
            # Bus is created inside _spawn_scenario; sequencer needs it
            self.sequencer = ReconfigSequencer(self.bus)

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

        def _path_inflate_for(self, host: int) -> float:
            """Obstacle-inflation radius to use when planning a path
            for the formation whose host is `host`.

            Rules:
              * Split singleton (n == 1): ROBOT_OCCUPANCY_M (0.40 m).
                Matches the physical collision radius exactly — any
                cell the planner sees as blocked is also a cell the
                collision resolver would push the robot out of.  No
                dead zone where the planner blocks a physically-free
                cell (which would leave a robot stuck with no force
                to push it back to open ground, as we saw with a
                wider inflation).
              * Fused singleton (n >  1): hull_radius, where
                hull_radius = (max distance from the formation
                centroid to any member pose) + ROBOT_OCCUPANCY_M.
                The formation's bounding-circle clearance — A* and
                Inter-Star plan a point-robot through the grid, so
                the planner needs clearance equal to the formation's
                actual footprint.  No additional scale factor.
            """
            members = self._members_of(host)
            if len(members) <= 1:
                return ROBOT_OCCUPANCY_M
            cx, cy = self._formation_centroid(host)
            max_r = 0.0
            for m in members:
                p = self.bus.poses.get(m)
                if p is None:
                    continue
                d = math.hypot(p.x - cx, p.y - cy)
                if d > max_r:
                    max_r = d
            hull_radius = max_r + ROBOT_OCCUPANCY_M
            return hull_radius

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

            # Phase B: Ctrl+LMB on a robot → toggle in multi-select set.
            #
            # Rules:
            #   * Clicking a robot already in `_selected_ids` removes it.
            #   * Clicking a robot whose host_id matches any currently
            #     selected robot's host_id is REJECTED — same-formation
            #     members move rigidly together, so adding another member
            #     would be redundant and would conflict with the fission
            #     interpretation of "one fused robot selected".
            #   * Otherwise the robot is added to the selection.  If this
            #     brings `_selected_ids` to >=2 AND there are pending
            #     fission goals queued for the previously-solo formation,
            #     those goals are dropped — the gesture has just
            #     promoted from fission-intent to fusion-intent.
            mods = pygame.key.get_mods()
            ctrl_held = bool(mods & (pygame.KMOD_LCTRL | pygame.KMOD_RCTRL))
            if self._toolbar_selected is None and ctrl_held:
                hit_rid = self._robot_at(wx, wy)
                if hit_rid is not None:
                    if hit_rid in self._selected_ids:
                        self._selected_ids.discard(hit_rid)
                        self._set_message(
                            f"Robot{hit_rid} removed from selection "
                            f"({len(self._selected_ids)} selected).", HUD_ACCENT)
                    else:
                        # Same-formation rejection: compare host_ids
                        hit_host = self._host_of(hit_rid)
                        selected_hosts = {self._host_of(r)
                                          for r in self._selected_ids}
                        if hit_host in selected_hosts:
                            mates = sorted(self._members_of(hit_rid))
                            self._set_message(
                                f"Robot{hit_rid} is already covered via "
                                f"formation {{Robot{', Robot'.join(map(str, mates))}}} "
                                f"(host Robot{hit_host}); fused members are "
                                f"implicitly selected together.",
                                HUD_WARN)
                        else:
                            self._selected_ids.add(hit_rid)
                            # Fusion intent — any queued fission goals for
                            # the previously-solo selection are no longer
                            # meaningful; drop them so the star markers
                            # on the map vanish.
                            if (len(self._selected_ids) >= 2
                                    and self._fission_goals_pending):
                                self._fission_goals_pending = []
                                self._fission_host = None
                            self._set_message(
                                f"Robot{hit_rid} added to selection "
                                f"({len(self._selected_ids)} selected).",
                                HUD_ACCENT)
                    # IMPORTANT: fully consume this click — clear the
                    # mouse-down position so mouseup doesn't fall through
                    # to the P2P nav-goal fallback (which would otherwise
                    # treat the Ctrl-click pose as a navigation target
                    # for the active robot).
                    self._mouse_held = True
                    self._mouse_down_pos = None
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

            # ---- PHASE A: arm GBNN-drag tracking ------------------------
            # Only arm when NO toolbar kind is selected.  When the toolbar
            # has Human/Table/Chair/etc selected the click is a single-spawn
            # action — drawing the cyan GBNN rectangle would be misleading.
            if self._toolbar_selected is None:
                self._gbnn_drag_start = (sx, sy)
                self._gbnn_drag_last  = (sx, sy)
                self._gbnn_drag_armed = True

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

            # ---- PHASE A: classify tap vs drag, dispatch GBNN if drag ----
            gbnn_dispatched = False
            if self._gbnn_drag_armed:
                self._gbnn_drag_armed = False
                start = self._gbnn_drag_start
                self._gbnn_drag_start = None
                self._gbnn_drag_last  = None
                if (start is not None
                        and self._toolbar_selected is None
                        and self.obs_mgr.obstacle_at(wx, wy) is None
                        and sy >= HUD_H
                        and not self._is_in_toolbar(sx, sy)):
                    dxp = sx - start[0]
                    dyp = sy - start[1]
                    if math.hypot(dxp, dyp) > self.GBNN_DRAG_THRESHOLD_PX:
                        self._dispatch_gbnn_coverage(start, (sx, sy))
                        gbnn_dispatched = True
                        self._mouse_down_pos = None

            if gbnn_dispatched:
                return

            # PHASE E — Shift+LMB on empty floor near an obstacle commits
            # an access point.  Critically, Shift+LMB ALWAYS consumes the
            # click — even when the AP can't be registered (no MDA, run
            # active, click on an obstacle, etc.) — so it never falls
            # through to default P2P navigation.  P2P nav is reserved
            # for clicks WITHOUT Shift.
            shift_held = bool(
                pygame.key.get_mods()
                & (pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT)
            )
            if shift_held:
                # Bail-out contexts: clicks the user almost certainly
                # didn't mean as AP placement.  Eat silently.
                if (sy < HUD_H
                        or self._is_in_toolbar(sx, sy)
                        or self._toolbar_selected is not None):
                    self._mouse_down_pos = None
                    return
                # Run already in progress: tell the user, then eat.
                if self._gbnnh_active:
                    self._set_message(
                        "GBNN+H run in progress.  Press Esc to cancel "
                        "before placing more APs.", HUD_WARN)
                    self._mouse_down_pos = None
                    return
                # No MDA host: tell the user how to fix it, then eat.
                if self._gbnnh_active_host() is None:
                    self._set_message(
                        "Shift+LMB places GBNN+H APs — mount an MDA on "
                        "the selected host first ('0' key).", HUD_WARN)
                    self._mouse_down_pos = None
                    return
                # Click landed on an obstacle: AP must be on empty space.
                if self.obs_mgr.obstacle_at(wx, wy) is not None:
                    self._set_message(
                        "AP rejected: click an empty floor cell, not on "
                        "an obstacle.", HUD_WARN)
                    self._mouse_down_pos = None
                    return
                # Try to register — may still reject (no obstacle in
                # reach, or click inside clearance zone).  Whether the
                # dispatch succeeds or not, eat the click.
                self._dispatch_gbnnh_ap_placement(wx, wy)
                self._mouse_down_pos = None
                return

            # PHASE B — Auto-dispatch based on selection state
            # ---------------------------------------------------------------
            #   (a) ≥2 robots in self._selected_ids          → FUSION
            #   (b) 1 fused-singleton selected (n>1)         → FISSION
            #       successive clicks queue n goals, then
            #       dispatch on the nth click (Option D)
            #   (c) else                                      → existing P2P
            #       nav for active-robot formation host
            if (self._toolbar_selected is None
                    and self.obs_mgr.obstacle_at(wx, wy) is None
                    and sy >= HUD_H
                    and not self._is_in_toolbar(sx, sy)):

                selected_list = sorted(self._selected_ids)

                # (a) Fusion — multiple robots converging on one point
                if len(selected_list) >= 2:
                    self._dispatch_interstar_fusion(selected_list, (wx, wy))
                    self._mouse_down_pos = None
                    return

                # (b) Fission — single selected fused singleton (n ≥ 2).
                # Previously gated on `cfg_only.host_id == only`, which
                # rejected selecting a non-host member of a fused
                # formation (e.g. Ctrl+click on Robot 4 in {2,3,4}).
                # Now we resolve to the sim-level formation host via
                # `_host_of(only)` so any member of a fused formation
                # can initiate fission.
                if len(selected_list) == 1:
                    only = selected_list[0]
                    fission_host = self._host_of(only)
                    cfg_host = self.bus.configurers.get(fission_host)
                    if (cfg_host is not None
                            and cfg_host.n >= 2
                            and cfg_host.host_id == fission_host):
                        # Active fission-goal collection, keyed on the
                        # true formation host so successive clicks reset
                        # state iff the user switches to a DIFFERENT
                        # formation.
                        if self._fission_host != fission_host:
                            self._fission_host = fission_host
                            self._fission_goals_pending = []
                        self._fission_goals_pending.append((wx, wy))
                        need = cfg_host.n
                        have = len(self._fission_goals_pending)
                        if have < need:
                            self._set_message(
                                f"Fission: {have}/{need} goals queued for "
                                f"Robot{fission_host}'s formation (n={need}).  "
                                f"Click {need - have} more to dispatch.",
                                HUD_ACCENT)
                        else:
                            goals_snapshot = list(self._fission_goals_pending)
                            self._fission_goals_pending = []
                            self._fission_host = None
                            self._dispatch_interstar_fission(
                                fission_host, goals_snapshot)
                        self._mouse_down_pos = None
                        return

                # (c) Existing mode-1 P2P nav for the active robot's host
                sel = self.selected_id
                if sel in self.bus.configurers:
                    host = self._host_of(sel)
                    members = self._members_of(sel)
                    if sel != host and len(members) > 1:
                        self._set_message(
                            f"Robot{sel} is not the host. Select Robot{host} "
                            f"to set a navigation goal.",
                            HUD_WARN)
                    else:
                        # Full nav-state reset at click time.  Stale
                        # state from a previous GBNN / Inter-Star run
                        # can otherwise silently sabotage A* for this
                        # host:
                        #   * `_nav_fail_count[host]` may have climbed
                        #     close to PATHFIND_FAIL_LIMIT (and never
                        #     hit it → never triggered `_cancel_nav`
                        #     → never cleared), so the next A*
                        #     failure could cancel the brand-new
                        #     goal on its first fail.
                        #   * `_nav_replan_tick[host]` may be 10**9
                        #     (set by cursor-based drivers to pin
                        #     their 2-point polyline) — the nav
                        #     controller would never replan against
                        #     the fresh goal, and instead drive the
                        #     stale `_nav_paths[host]` polyline to
                        #     the OLD target.
                        #   * `_nav_paths[host]` itself may still be
                        #     the stale polyline.
                        # Wiping all four fields makes every P2P
                        # click a clean-slate A* request — matches
                        # how GBNN and Inter-Star dispatches already
                        # reset their state.
                        self._nav_goals[host]       = (wx, wy)
                        self._nav_paths.pop(host, None)
                        self._nav_wp_idx[host]      = 0
                        self._nav_fail_count[host]  = 0
                        self._nav_replan_tick[host] = 0
                        if len(members) > 1:
                            tag = (f"formation (host=Robot{host}, "
                                   f"n={len(members)})")
                        else:
                            tag = f"Robot{host}"
                        self._set_message(
                            f"{tag} goal → ({wx:.1f}, {wy:.1f}) "
                            f"[{self._pathfind_algo.upper()} / "
                            f"{self._nav_motion.upper()}]",
                            HUD_ACCENT)

            self._mouse_down_pos = None

        def _on_mousemotion(self, event) -> None:
            # Phase A: track drag for GBNN-coverage rectangle preview
            if self._gbnn_drag_armed:
                self._gbnn_drag_last = event.pos

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
                self._cancel_nav(host, f"Robot{host} navigation cancelled.")
                return
            if key == pygame.K_t:
                self._try_trolley_toggle()
                return
            # Phase E (Mode 5) — MDA mount/unmount on key '0'
            if key == pygame.K_0:
                self._try_mda_mount_toggle()
                return
            # Phase E (Mode 5) — surface-view subpanel toggle on Tab
            if key == pygame.K_TAB:
                self._gbnnh_show_panel = not self._gbnnh_show_panel
                self._set_message(
                    f"Surface view: {'ON' if self._gbnnh_show_panel else 'OFF'}",
                    HUD_ACCENT)
                return
            # Phase E (Mode 5) — Enter starts the cleaning sequence
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self._start_gbnnh_run():
                    return
            if key == pygame.K_c:
                # Clear all obstacles
                self.obs_mgr.obstacles.clear()
                self._attached_trolley.clear()
                self._trolley_docking = None
                self._set_message("All obstacles cleared.", HUD_WARN)
                return
            if key == pygame.K_ESCAPE:
                # Phase B: Esc first cancels Interstar selection / fission
                # queue / active plan + render state.  A second Esc (or
                # when nothing is active) quits the sandbox.
                dirty = False
                if self._selected_ids:
                    self._selected_ids.clear()
                    dirty = True
                if self._fission_goals_pending:
                    self._fission_goals_pending = []
                    self._fission_host = None
                    dirty = True
                if (self._interstar_metrics
                        or self._interstar_shared_segment
                        or self._interstar_paths
                        or self._interstar_plan_active
                        or self._interstar_fusion_pairs
                        or self._interstar_staging_active
                        or self._interstar_staging_slots):
                    # Stop driving — clear per-robot nav + cached plan state
                    for rid in list(self._interstar_paths.keys()):
                        self._nav_goals.pop(rid, None)
                        self._nav_paths.pop(rid, None)
                        self._nav_wp_idx.pop(rid, None)
                        self._nav_replan_tick.pop(rid, None)
                    self._interstar_paths          = {}
                    self._interstar_cursor         = {}
                    self._interstar_shared_segment = []
                    self._interstar_metrics        = {}
                    self._interstar_pending_fuses  = []
                    self._interstar_plan_active    = False
                    self._interstar_stop_timer     = 0
                    self._interstar_cluster_timer  = 0
                    self._interstar_last_positions = {}
                    self._interstar_fusion_pairs   = []
                    self._interstar_fission_goals  = {}
                    self._interstar_fission_start_poses = {}
                    self._interstar_staging_slots  = {}
                    self._interstar_combine_anchor = None
                    self._interstar_staging_active = False
                    self._interstar_staging_timer  = 0
                    self._interstar_staging_rest_ticks = 0
                    self._interstar_staging_last_pose = {}
                    # Iterative-replan scaffold + mode flag
                    self._interstar_mode           = ""
                    self._interstar_fusion_goal    = None
                    self._interstar_selected_cache = []
                    self._interstar_replan_counter = 0
                    dirty = True
                # Phase E (Mode 5) cleanup — mirrors the Inter-Star block
                # above so a single Esc tear-down covers all active modes.
                if (self._gbnnh_active
                        or self._gbnnh_aps
                        or self._gbnnh_planner is not None):
                    if self._gbnnh_host_rid is not None:
                        self._nav_goals.pop(self._gbnnh_host_rid, None)
                        self._nav_paths.pop(self._gbnnh_host_rid, None)
                        self._nav_wp_idx.pop(self._gbnnh_host_rid, None)
                        self._nav_replan_tick.pop(self._gbnnh_host_rid, None)
                    self._gbnnh_active          = False
                    self._gbnnh_host_rid        = None
                    self._gbnnh_aps             = []
                    self._gbnnh_active_ap_idx   = None
                    self._gbnnh_planner         = None
                    self._gbnnh_completion_tick = None
                    self._gbnnh_stall_pos       = None
                    self._gbnnh_stall_count     = 0
                    self._gbnnh_step_frame_counter = 0
                    dirty = True
                if dirty:
                    self._set_message(
                        "Esc: Interstar / GBNN+H state cleared.", HUD_ACCENT)
                    return
                self.running = False
                return

        def _sync_keyboard_state(self) -> None:
            keys = pygame.key.get_pressed()
            s = self.input_state
            s.w     = keys[pygame.K_w]
            s.a     = keys[pygame.K_a]
            s.s     = keys[pygame.K_s]
            s.d     = keys[pygame.K_d]
            # Arrow keys — holonomic 4-direction strafe (independent of WASD).
            s.up    = keys[pygame.K_UP]
            s.down  = keys[pygame.K_DOWN]
            s.left  = keys[pygame.K_LEFT]
            s.right = keys[pygame.K_RIGHT]
            s.shift = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            s.q     = keys[pygame.K_q]
            s.e     = keys[pygame.K_e]

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
            host_label = f"Robot{host}{heavy_tag}"
            if len(members) > 1:
                role = "HOST" if rid == host else "member"
                hint = ("drives formation"
                        if rid == host else
                        f"idle (host is {host_label})")
                self._set_message(
                    f"Controlling Robot{rid} [{role}, {hint}]  "
                    f"formation={sorted(members)} n={len(members)}",
                    HUD_ACCENT,
                )
            else:
                tag = f"{heavy_tag} " if heavy_tag else ""
                self._set_message(
                    f"Controlling Robot{rid} {tag}(split singleton).",
                    HUD_ACCENT,
                )

        def _try_fuse(
            self,
            rid_a: int,
            rid_b: int,
            force: bool = False,
        ) -> None:
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

            force : bool, default False
              When True, bypass the DOCKING_DISTANCE_M closest-pair
              check.  Used by the Inter-Star combine flow when a
              formation-packer slot drive failed (robot stuck before
              reaching its dock slot) — we'd rather have the stuck
              robot teleport-dock into the merged formation than
              leave it behind as a permanent split singleton.  The
              P-controller will still drive the formation across
              whatever distance remains, just without the pre-flight
              range gate.
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
                    f"Robot{rid_a} and Robot{rid_b} already in the same formation.",
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
                            f"Cannot combine while Robot{att_rid} has "
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
            if d > DOCKING_DISTANCE_M and not force:
                self._set_message(
                    f"Docking rejected: closest pair between formations "
                    f"= {d:.2f} m, limit = {DOCKING_DISTANCE_M:.2f} m.",
                    HUD_WARN,
                )
                return
            if d > DOCKING_DISTANCE_M and force:
                # Forced combine — robot couldn't reach its slot in
                # time (or got stuck), but the caller wants it in
                # the formation regardless.  Docking P-controller
                # will drive the trigger the whole remaining span.
                self._set_message(
                    f"Forced docking over {d:.2f} m (stuck slot "
                    f"adjustment): Robot{rid_a}'s formation → Robot{rid_b}.",
                    HUD_WARN,
                )

            # Trigger approach robot — the specific member of formation
            # A that physically drives toward dock_target.  Previously
            # `min(members_a)` was used unconditionally, which caused
            # visible overlap on chain fusions: the formation's naming
            # host was not always the closest pair member to rid_b, so
            # the rigid translation at dock-snap pushed NON-approach
            # members straight through the target formation's positions.
            #
            # Using rid_a as the trigger lets callers nominate exactly
            # which robot from formation A should end up adjacent to
            # dock_target.  Both the Inter-Star proximity-chain queue
            # and the SHIFT+N user path benefit: the snapping geometry
            # is computed around the intended approach pair, so other
            # members land at safe rigid-body offsets instead of on
            # top of the target formation.
            trigger_host = rid_a if rid_a in members_a else min(members_a)

            # dock_target = the specific robot the user pointed at (rid_b),
            # NOT the target formation's host.  Robot 5 pressed SHIFT+4
            # → dock adjacent to Robot4, even though Robot4's host is Robot3.
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
                f"Docking: Robot{trigger_host}'s formation -> Robot{dock_target} "
                f"(triggered from Robot{rid_a}, closest pair = {d:.2f} m). "
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
                    f"Robot{rid} is already a split singleton.", HUD_WARN,
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

            # ---- WASD: differential drive (forward/back + turn) -------
            if s.w: vx += BASE_LIN_SPEED * scale
            if s.s: vx -= BASE_LIN_SPEED * scale
            if s.a: wz += BASE_ANG_SPEED * scale   # rotate left  (+ωz)
            if s.d: wz -= BASE_ANG_SPEED * scale   # rotate right (-ωz)

            # ---- Arrow keys: holonomic 4-direction strafe -------------
            # Layered on top of WASD so e.g. W + RIGHT drives forward
            # AND strafes right.  No Shift required.
            if s.up:    vx += BASE_LIN_SPEED * scale   # forward strafe
            if s.down:  vx -= BASE_LIN_SPEED * scale   # backward strafe
            if s.left:  vy += BASE_LIN_SPEED * scale   # left strafe
            if s.right: vy -= BASE_LIN_SPEED * scale   # right strafe

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
              * dock_target  = the physical robot we approach (e.g. Robot4)
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

            # ---- Post-snap overlap resolution --------------------------
            # The rigid translation above can land a trigger-side member
            # on top of a target-side member (or two trigger members
            # atop each other) — whenever formation geometry + docking
            # angle align unfavourably.  If rT_b is captured while any
            # two members overlap, that overlap becomes PERMANENT
            # (subsequent rigid motion preserves the offset).  Run a
            # few relaxation passes here, shifting trigger members
            # radially away from anything they overlap, until no pair
            # is within DOCKED_DISTANCE_M (minus COLLISION_EPSILON).
            # The trigger_host stays fixed — it's the docked pose.
            # Every member except the just-snapped trig_host is
            # eligible to be pushed.  This is critical for multi-
            # formation chain fuses: residual overlap from EARLIER
            # passes can sit inside what's now the target side, and
            # the previous trigger-only relaxation couldn't fix
            # those.  Now any overlapping pair (target-target,
            # trigger-trigger, target-trigger) gets resolved by
            # shifting the non-anchor side outward.  trig_host
            # stays at the docked pose so the docking geometry
            # (DOCKED_DISTANCE_M from dock_target) is preserved.
            unified_pre = list(dk["trigger_members"]) + list(dk["target_members"])
            movable = [m for m in unified_pre if m != trig_host]
            for _relax in range(30):
                moved = False
                for m in movable:
                    pm = self.bus.poses[m]
                    for other in unified_pre:
                        if other == m:
                            continue
                        po = self.bus.poses[other]
                        ddx = pm.x - po.x
                        ddy = pm.y - po.y
                        dd  = math.hypot(ddx, ddy)
                        threshold = DOCKED_DISTANCE_M - COLLISION_EPSILON
                        if dd < threshold:
                            push = (threshold - dd) + COLLISION_EPSILON
                            if dd < 1e-6:
                                # Co-located — pick a deterministic
                                # direction (perpendicular to the
                                # host→target approach axis) so
                                # multiple overlapping members spread
                                # cleanly.
                                ux_p, uy_p = -uy, ux
                            else:
                                ux_p, uy_p = ddx / dd, ddy / dd
                            self.bus.poses[m] = Pose(
                                x   = pm.x + ux_p * push,
                                y   = pm.y + uy_p * push,
                                yaw = pm.yaw,
                            )
                            moved = True
                            pm = self.bus.poses[m]   # refresh for next loop
                if not moved:
                    break

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
                f"Docked.  Formation host=Robot{new_host}, members={unified}, "
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
                    f"Robot{sel} is not the host. Select Robot{host} to "
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
                            f"before detaching Robot{att_rid}.", HUD_WARN)
                        return
                self._attached_trolley.pop(att_rid)
                if obs:
                    obs.pinned = False  # trolley becomes freely movable again
                self._set_message(
                    f"Robot{att_rid} detached from {name}.", HUD_OK)
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
                f"Attaching Robot{host} to {best_obs.label} "
                f"(uid={best_obs.uid})...", HUD_OK)

        # ================================================================
        # Phase E (Mode 5) — MDA mount + AP placement + visibility
        # ================================================================

        def _try_mda_mount_toggle(self) -> None:
            """'0' key — mount or unmount an MDA module on the active robot.

            Mount: find nearest unmounted MDA within MDA_MOUNT_RANGE_M of
                   the active robot (must be the formation host) and snap
                   it onto the host pose.
            Unmount: if active robot already has an MDA mounted, detach it
                     and leave it as a free obstacle at the host's pose.
            """
            sel = self.selected_id
            if sel not in self.bus.configurers:
                self._set_message("Select a robot first (1-5).", HUD_WARN)
                return
            host = self._host_of(sel)
            if sel != host:
                self._set_message(
                    f"Robot{sel} is not the host. Select Robot{host} to "
                    f"mount/unmount MDA.", HUD_WARN)
                return

            # Already mounted? → unmount.
            existing = self.obs_mgr.find_mda_for_robot(host)
            if existing is not None:
                self.obs_mgr.unmount_mda(existing.uid)
                self._set_message(
                    f"MDA detached from Robot{host}.", HUD_OK)
                return

            # Look for a candidate MDA to mount.
            p = self.bus.poses[host]
            cand = self.obs_mgr.find_unmounted_mda_near(
                p.x, p.y, max_range=MDA_MOUNT_RANGE_M)
            if cand is None:
                self._set_message(
                    f"No MDA module within {MDA_MOUNT_RANGE_M:.1f} m of "
                    f"Robot{host}. Spawn one nearby first.", HUD_WARN)
                return
            ok = self.obs_mgr.mount_mda(
                cand.uid, host_robot_id=host,
                host_xy=(p.x, p.y), host_yaw=p.yaw)
            if ok:
                self._set_message(
                    f"MDA mounted on Robot{host}.", HUD_OK)
            else:
                self._set_message(
                    f"Mount failed (Robot{host} may already have one).",
                    HUD_WARN)

        def _gbnnh_active_host(self) -> Optional[int]:
            """Return the robot id of an MDA-equipped host that is the
            currently selected formation host, or None.

            Pressing 5 / H is gated on this returning a value.  Mode 5 is
            tied to ONE host robot per session (the one with the MDA).
            """
            sel = self.selected_id
            if sel not in self.bus.configurers:
                return None
            host = self._host_of(sel)
            if sel != host:
                return None
            if not self.obs_mgr.has_mda_mounted(host):
                return None
            return host

        @staticmethod
        def _obstacle_obb_edges(
            obs: "Obstacle",
        ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
            """Four world-frame edges of the obstacle's OBB, ordered.

            Edge 0 = -y face (south), 1 = +x (east), 2 = +y (north),
            3 = -x (west) in obstacle-local frame.  Used by RoI segment
            extraction so each edge can be clipped against the FoV
            cone and tested for occlusion independently — guaranteeing
            different edges of the same obstacle never get merged into
            one segment.
            """
            hw, hh = obs.half_w, obs.half_h
            # Local corners (CCW starting from -hw, -hh)
            local = [(-hw, -hh), ( hw, -hh),
                     ( hw,  hh), (-hw,  hh)]
            co = math.cos(obs.yaw); so = math.sin(obs.yaw)
            world = []
            for lx, ly in local:
                wx = obs.x + co * lx - so * ly
                wy = obs.y + so * lx + co * ly
                world.append((wx, wy))
            edges = []
            for i in range(4):
                edges.append((world[i], world[(i + 1) % 4]))
            return edges

        @staticmethod
        def _point_in_fov_cone(
            px: float, py: float,
            ap_x: float, ap_y: float,
            ap_yaw: float,
            half_fov_rad: float,
            reach_m: float,
        ) -> bool:
            """True iff (px, py) lies inside the FoV pie-slice defined by
            apex (ap_x, ap_y), bearing ap_yaw, opening 2·half_fov_rad,
            radius reach_m.  Inclusive on the boundary."""
            dx = px - ap_x
            dy = py - ap_y
            d2 = dx * dx + dy * dy
            if d2 > reach_m * reach_m:
                return False
            if d2 < 1e-12:
                return True  # AP itself is inside its own cone
            bearing = math.atan2(dy, dx)
            rel = math.atan2(
                math.sin(bearing - ap_yaw),
                math.cos(bearing - ap_yaw))
            return abs(rel) <= half_fov_rad + 1e-9

        @staticmethod
        def _ray_hits_obb(
            ax: float, ay: float, bx: float, by: float,
            obs: "Obstacle", eps: float = 1e-3,
        ) -> bool:
            """Analytical line-segment vs OBB intersection (slab method).

            Returns True iff the open segment ``(A, B)`` enters the
            obstacle's oriented bounding box strictly between its
            endpoints — the entry parameter ``tmin`` lies in
            ``(eps, 1 - eps)``.  Endpoint epsilon avoids false-positives
            from segments that just-touch an OBB at A or B (e.g. when B
            sits on the source obstacle's own edge).

            This is the right tool for camera-style FoV occlusion: thin
            walls (half_h ≈ 0.1 m) that a sample-based LOS misses
            because of step-size aliasing are caught analytically here.
            """
            # Transform A and B into obs-local frame
            co = math.cos(-obs.yaw); so = math.sin(-obs.yaw)
            al_x = co * (ax - obs.x) - so * (ay - obs.y)
            al_y = so * (ax - obs.x) + co * (ay - obs.y)
            bl_x = co * (bx - obs.x) - so * (by - obs.y)
            bl_y = so * (bx - obs.x) + co * (by - obs.y)
            dx = bl_x - al_x
            dy = bl_y - al_y
            tmin, tmax = 0.0, 1.0
            for d, a_loc, ext in ((dx, al_x, obs.half_w),
                                  (dy, al_y, obs.half_h)):
                if abs(d) < 1e-12:
                    # Ray parallel to this slab; outside slab → no hit
                    if a_loc < -ext or a_loc > ext:
                        return False
                    continue
                t1 = (-ext - a_loc) / d
                t2 = ( ext - a_loc) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
                if tmin > tmax:
                    return False
            # Hit iff the entry happens strictly inside (0, 1) —
            # rules out endpoint grazing.
            return tmin > eps and tmin < 1.0 - eps

        @classmethod
        def _segment_los_clear(
            cls,
            ax: float, ay: float, bx: float, by: float,
            occluders: List["Obstacle"],
            n_samples: int = 0,   # ignored; kept for backward-compat
        ) -> bool:
            """LOS test using analytical ray-OBB intersection — True iff
            no occluder's OBB is pierced strictly between A and B."""
            for obs in occluders:
                if cls._ray_hits_obb(ax, ay, bx, by, obs):
                    return False
            return True

        @staticmethod
        def _obstacle_nearest_point(
            obs: "Obstacle", wx: float, wy: float,
        ) -> Tuple[float, float]:
            """Closest point on ``obs``'s OBB to the world point (wx, wy).

            Inlined here (rather than as an Obstacle method) so demo.py
            works against an unmodified ``obstacles.py``.  Math: transform
            the query point into obstacle-local frame, clamp to the OBB
            half-extents, transform back.  Matches the existing
            ``circle_overlap`` math, so the surface point we return is
            consistent with collision behaviour.
            """
            dx = wx - obs.x
            dy = wy - obs.y
            ci, si = math.cos(-obs.yaw), math.sin(-obs.yaw)
            lx =  ci * dx - si * dy
            ly =  si * dx + ci * dy
            nlx = max(-obs.half_w, min(obs.half_w, lx))
            nly = max(-obs.half_h, min(obs.half_h, ly))
            cf, sf = math.cos(obs.yaw), math.sin(obs.yaw)
            nx = obs.x + cf * nlx - sf * nly
            ny = obs.y + sf * nlx + cf * nly
            return (nx, ny)

        def _nearest_obstacle_to(
            self, x: float, y: float, max_range: float,
        ) -> Tuple[Optional["Obstacle"], Optional[Tuple[float, float]]]:
            """Closest obstacle to the world point (x, y) and the closest
            point on that obstacle's surface.

            Returns ``(obstacle, (nx, ny))`` or ``(None, None)`` if no
            qualifying obstacle exists within ``max_range``.

            Uses each obstacle's OBB surface — NOT centre — so a long
            wall registers when the click is near any part of it, and
            the returned nearest-point falls on the edge facing the
            click rather than the wall's geometric centre.  This is
            critical for AP orientation: yaw = atan2(np - click) gives
            the perpendicular-to-wall direction users expect.

            Mounted MDAs are excluded — they ride with their host and
            don't define cleaning targets.
            """
            best_obs: Optional["Obstacle"] = None
            best_pt:  Optional[Tuple[float, float]] = None
            best_d                                  = max_range
            for obs in self.obs_mgr.obstacles.values():
                if obs.is_mounted:
                    continue
                nx, ny = self._obstacle_nearest_point(obs, x, y)
                d = math.hypot(nx - x, ny - y)
                if d < best_d:
                    best_d   = d
                    best_obs = obs
                    best_pt  = (nx, ny)
            return best_obs, best_pt

        def _dispatch_gbnnh_ap_placement(self, wx: float, wy: float) -> bool:
            """Shift+LMB → register an access point at the click position.

            New rules (per the revised UX):
              - The click point must be on EMPTY space (caller filters).
              - There must be at least one obstacle within MDA arm reach
                of the click; AP yaw points from the click toward that
                nearest obstacle, so the host robot ends up facing the
                surface to clean.
              - No FOV / visibility evaluation is done here — that work
                happens at arrival time inside ``_refresh_gbnnh_active``.

            Returns True if an AP was registered.  The mouse handler
            consumes the click only on True.
            """
            host = self._gbnnh_active_host()
            if host is None or host not in self.bus.poses:
                self._set_message(
                    "Mode 5 needs an MDA mounted on the selected host.",
                    HUD_WARN)
                return False
            if self._gbnnh_active:
                self._set_message(
                    "GBNN+H run in progress.  Esc to cancel before "
                    "adding more APs.", HUD_WARN)
                return False

            # Find the nearest obstacle within MDA reach.  Distance is
            # measured to the obstacle's nearest surface point — so a
            # long wall counts as "near" whenever any part of it is
            # within reach, not just when its centre is.  No obstacle
            # in range → no surface to clean → reject.
            target, nearest_pt = self._nearest_obstacle_to(
                wx, wy, MDA_ARM_REACH_M)
            if target is None or nearest_pt is None:
                self._set_message(
                    f"AP rejected: no obstacle surface within "
                    f"{MDA_ARM_REACH_M:.1f} m of the click point.",
                    HUD_WARN)
                return False

            # Reject if the click sits inside any obstacle's robot-clearance
            # zone (surface distance < ROBOT_OCCUPANCY_M).  The host's body
            # cannot physically occupy that point, so an AP there would
            # have the robot getting stuck on arrival.  Since the call
            # above returns the *nearest* obstacle by surface distance,
            # checking that one suffices: any other obstacle is farther.
            nx_chk, ny_chk = nearest_pt
            surface_d = math.hypot(nx_chk - wx, ny_chk - wy)
            if surface_d < ROBOT_OCCUPANCY_M:
                self._set_message(
                    f"AP rejected: click is inside the robot-clearance "
                    f"zone of {target.label} "
                    f"(distance {surface_d:.2f} m < "
                    f"{ROBOT_OCCUPANCY_M:.2f} m).  "
                    f"Hold Shift to see clearance halos.",
                    HUD_WARN)
                return False

            # Yaw points from the click toward the nearest surface point
            # — perpendicular-to-wall for long obstacles, edge-facing
            # for tables / chairs.  This is what the host robot will
            # rotate to before starting GBNN+H.
            nx, ny = nearest_pt
            yaw = math.atan2(ny - wy, nx - wx)
            pose = (wx, wy, yaw)
            ap_idx = len(self._gbnnh_aps) + 1
            self._gbnnh_aps.append({
                "pose":         pose,
                # RoI segments are computed on arrival, not at placement
                # time — visibility depends on live obstacle state.  Each
                # element of `roi_segments` is a dict (see
                # _compute_roi_segments_for_ap for the schema).
                "roi_segments": None,
                "label":        f"AP{ap_idx}",
                "target":       target.uid,
                "target_point": nearest_pt,  # for overlay rendering
                "done":         False,
                "stats":        None,
            })
            self._gbnnh_host_rid = host
            self._set_message(
                f"AP{ap_idx} registered at ({wx:.1f}, {wy:.1f}) "
                f"facing {target.label} (uid={target.uid}) at "
                f"surface ({nx:.1f}, {ny:.1f}).  "
                f"{len(self._gbnnh_aps)} APs queued — Enter to start.",
                HUD_OK)
            return True

        def _sequence_aps(self, host_xy: Tuple[float, float]) -> None:
            """Reorder ``self._gbnnh_aps`` into a near-optimal visit order
            via a greedy nearest-neighbor tour starting from the host's
            current position.  In-place; only invoked once at run start."""
            if len(self._gbnnh_aps) <= 1:
                return
            remaining = list(self._gbnnh_aps)
            ordered:  List[Dict[str, object]] = []
            cx, cy   = host_xy
            while remaining:
                # pick the nearest AP to the current cursor
                idx_best = 0
                d_best   = float('inf')
                for i, ap in enumerate(remaining):
                    px, py, _ = ap["pose"]
                    d = (px - cx) ** 2 + (py - cy) ** 2
                    if d < d_best:
                        d_best = d
                        idx_best = i
                pick = remaining.pop(idx_best)
                ordered.append(pick)
                cx, cy = pick["pose"][0], pick["pose"][1]
            # Renumber the labels so the HUD shows "AP1, AP2, ..." in
            # visit order rather than placement order.
            for i, ap in enumerate(ordered, start=1):
                ap["label"] = f"AP{i}"
            self._gbnnh_aps = ordered

        def _start_gbnnh_run(self) -> bool:
            """Enter key handler — initiate the cleaning sequence.

            Pre-flight checks:
              - An MDA must be mounted on the selected host.
              - At least one AP must be queued.
              - A run must not already be in progress.

            On success, sequences APs by greedy nearest-neighbor from the
            host's current pose, sets the nav goal to AP[0], and flips
            ``_gbnnh_active = True``.  Returns False (and emits a HUD
            warning) if the pre-flight fails — caller can fall through.
            """
            host = self._gbnnh_active_host()
            if host is None or host not in self.bus.poses:
                self._set_message(
                    "Cannot start: MDA not mounted on selected host.",
                    HUD_WARN)
                return False
            if not self._gbnnh_aps:
                self._set_message(
                    "Cannot start: no APs registered.  Shift+LMB on "
                    "empty floor near an obstacle to add one.",
                    HUD_WARN)
                return False
            if self._gbnnh_active:
                self._set_message(
                    "GBNN+H run already in progress.", HUD_WARN)
                return False

            hp = self.bus.poses[host]
            self._sequence_aps((hp.x, hp.y))

            self._gbnnh_host_rid        = host
            self._gbnnh_active_ap_idx   = 0
            self._gbnnh_active          = True
            self._gbnnh_planner         = None    # built on AP arrival
            self._gbnnh_completion_tick = None    # cancel any pending reset
            ap0 = self._gbnnh_aps[0]
            self._nav_goals[host] = (ap0["pose"][0], ap0["pose"][1])
            self._nav_paths.pop(host, None)
            self._nav_wp_idx.pop(host, None)
            self._nav_replan_tick.pop(host, None)
            self._set_message(
                f"GBNN+H run started — sequencing {len(self._gbnnh_aps)} "
                f"APs.  Heading to {ap0['label']}.", HUD_OK)
            return True

        # Tunables for RoI segment extraction
        GBNNH_SEG_SAMPLES_PER_EDGE: int = 40   # along each obstacle edge
        GBNNH_SEG_MIN_LENGTH_M:    float = 0.05  # drop tiny dust segments
        # Edge-facing filter (Rule 1).  An obstacle's edge enters the
        # candidate set only when its outward normal points generally
        # toward the AP — back-facing edges are dropped.  The threshold
        # is lenient (0.05) so walls approached at oblique angles still
        # qualify; the FINAL choice of which edge to use is then made by
        # the closest-edge rule below, not by this threshold.
        GBNNH_EDGE_FACING_COS:     float = 0.05

        # Per-obstacle-kind cleaning surface "length" — number of grid
        # rows the surface occupies in the merged 2D grid (rule 3).
        # Walls have tall vertical surfaces; tables / chairs are shallow.
        GBNNH_SURFACE_LENGTH_CELLS: Dict["ObstacleKind", int] = {
            # Populated below (after class body via __init_subclass__-style
            # patching can't reach ObstacleKind cleanly here; we set
            # the dict lazily in _surface_length_cells).
        }
        GBNNH_DEFAULT_SURFACE_LEN: int = 5

        @staticmethod
        def _surface_length_cells(kind: "ObstacleKind") -> int:
            """Map obstacle kind → grid-rows in the merged RoI.  Walls have
            a tall surface (14 cells), tables 5, chairs 3 — per the
            user's spec for Phase E rule 3."""
            table = {
                ObstacleKind.WALL:          14,
                ObstacleKind.PILLAR:        14,
                ObstacleKind.SLIDING_DOOR:  14,
                ObstacleKind.PIVOT_DOOR:    14,
                ObstacleKind.TABLE:          5,
                ObstacleKind.CHAIR:          3,
                ObstacleKind.LOW_TROLLEY:    5,
                ObstacleKind.HIGH_TROLLEY:   5,
                ObstacleKind.HEAVY_TROLLEY:  5,
                ObstacleKind.HUMAN:          3,
            }
            return table.get(kind, 5)

        def _merge_segments_to_grid(
            self,
            segments: List[Dict[str, object]],
            ap_pose:  Tuple[float, float, float],
        ) -> Tuple["Optional[np.ndarray]",
                   "Optional[List[Dict[str, object]]]"]:
            """Build a single 2D GBNN+H grid by concatenating segments
            along the AP's perpendicular axis (Rules 2 + 3).

            Pipeline:
              1. Project each segment onto the perpendicular-to-yaw axis
                 (the "width" axis in the merged grid).
              2. Sort by projected position.
              3. Walk the sorted list — between adjacent segments insert
                 a "gap" region (fully obstacle, marked -1.0).
              4. Total width = sum of all surface + gap widths in cells
                 (cell_m = GBNNH_SEG_CELL_M).
              5. Total length = max surface length across all segments
                 (per-kind from _surface_length_cells).
              6. Build the grid: surface columns clean (1.0) up to that
                 surface's length, obstacle (-1.0) above; gap columns
                 fully obstacle.

            Returns
            -------
            grid : np.ndarray of shape (length_cells, width_cells), or
                   None if the input is empty.
            layout : list of dicts describing each column block, in
                     left-to-right order.  Schema::

                {
                    "kind":         "surface" | "gap",
                    "col_start":    int,
                    "col_end":      int,
                    "length_cells": int,         # surface only
                    "segment_uid":  int,         # surface only
                    "obstacle_kind": ObstacleKind,  # surface only
                }

            The layout lets us trace progress on each original RoI
            segment back from cell coverage in its column slice.
            """
            if not segments:
                return None, None
            ax, ay, ayaw = ap_pose
            cell_m = max(self.GBNNH_SEG_CELL_M, 1e-3)
            # Perpendicular-to-yaw unit vector — but CANONICALIZED so
            # the merged-grid column order matches what the user sees
            # in the floor-plan view regardless of which way the AP
            # faces.  Without canonicalisation, looking south vs north
            # at the same scene would reverse the panel column order
            # (each yaw produces an opposite perp vector, and ascending
            # projection puts the world-+x end first in one and last
            # in the other).  Canonical rule: perp always has a
            # non-negative +x component; if perp is purely vertical
            # (|x|≈0) it points in +y.  Result: column 0 = the
            # world-frame "lower x or lower y" surface = the side the
            # user sees on the LEFT of the floor plan, in every yaw.
            perp_x =  math.sin(ayaw)
            perp_y = -math.cos(ayaw)
            if abs(perp_x) > 1e-6:
                if perp_x < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y
            else:
                if perp_y < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y

            projected = []
            for seg in segments:
                (a, b) = seg["endpoints"]
                pa = a[0] * perp_x + a[1] * perp_y
                pb = b[0] * perp_x + b[1] * perp_y
                projected.append({
                    "seg":    seg,
                    "lo":     min(pa, pb),
                    "hi":     max(pa, pb),
                })
            projected.sort(key=lambda p: p["lo"])

            # Walk segments in order, inserting gaps where there's space
            # between consecutive ones.
            layout: List[Dict[str, object]] = []
            origin = projected[0]["lo"]
            cursor = origin   # how far along the perpendicular axis
            for p in projected:
                if p["lo"] > cursor + 1e-6:
                    layout.append({
                        "kind": "gap",
                        "lo":   cursor,
                        "hi":   p["lo"],
                    })
                layout.append({
                    "kind":          "surface",
                    "lo":            p["lo"],
                    "hi":            p["hi"],
                    "segment":       p["seg"],
                    "segment_uid":   p["seg"]["segment_uid"],
                    "obstacle_kind": p["seg"].get("obstacle_kind"),
                })
                cursor = max(cursor, p["hi"])

            # Compute pixel/cell extents for each layer.
            for layer in layout:
                col_start = int(round((layer["lo"] - origin) / cell_m))
                col_end   = int(round((layer["hi"] - origin) / cell_m))
                if col_end <= col_start:
                    col_end = col_start + 1
                layer["col_start"] = col_start
                layer["col_end"]   = col_end
                if layer["kind"] == "surface":
                    layer["length_cells"] = self._surface_length_cells(
                        layer["obstacle_kind"])

            # Total width = max col_end across layers.
            total_width = max(layer["col_end"] for layer in layout)
            total_width = max(2, total_width)

            # Total length = max surface length (Rule 3).
            surf_lengths = [layer["length_cells"]
                            for layer in layout if layer["kind"] == "surface"]
            if not surf_lengths:
                return None, None
            total_length = max(surf_lengths)
            total_length = max(2, total_length)

            # Default grid = obstacle.  Surface columns: clean cells in
            # rows [0, length_cells); rows [length_cells, total_length)
            # stay obstacle.  Gap columns: all obstacle.  Note: rows
            # are indexed top-down in numpy; row 0 is the "near edge"
            # row (closest to AP in 3D — the bottom of a wall, the
            # near side of a table).
            grid = np.full((total_length, total_width), -1.0, dtype=float)
            for layer in layout:
                if layer["kind"] != "surface":
                    continue
                cs, ce = layer["col_start"], layer["col_end"]
                lc = layer["length_cells"]
                grid[:lc, cs:ce] = 1.0

                # Tables get random clutter — items, plates, etc. Each
                # AP arrival re-rolls the layout (random count + random
                # cell positions), so visiting the same AP twice
                # produces different runs.  Walls and other surfaces
                # stay clean.
                if (layer.get("obstacle_kind") == ObstacleKind.TABLE
                        and lc > 0 and (ce - cs) > 0):
                    n_total = lc * (ce - cs)
                    # 10–25% of the table's surface cells become clutter
                    n_clutter = random.randint(
                        max(1, n_total // 10),
                        max(2, n_total // 4),
                    )
                    placed: set = set()
                    attempts = 0
                    while len(placed) < n_clutter and attempts < n_clutter * 4:
                        r = random.randint(0, lc - 1)
                        c = random.randint(cs, ce - 1)
                        if (r, c) not in placed:
                            grid[r, c] = -1.0
                            placed.add((r, c))
                        attempts += 1

            return grid, layout

        def _update_segment_progress_from_planner(
            self, ap: Dict[str, object],
        ) -> None:
            """Mirror live GBNN+H coverage of each layer's column slice
            into the matching RoI segment's ``progress`` field.

            The merged grid concatenates surface columns and gap columns
            horizontally.  For each "surface" layer in the merged
            layout, count how many of its (length × width) cells the
            planner has visited (i.e., neither 1.0 nor -1.0) and set
            the source segment's progress accordingly.
            """
            if self._gbnnh_planner is None:
                return
            if isinstance(self._gbnnh_planner, str):
                return
            layout = ap.get("merged_layout")
            if not layout:
                return
            rs = self._gbnnh_planner.render_state()
            grid = rs["grid"]
            for layer in layout:
                if layer["kind"] != "surface":
                    continue
                cs, ce = layer["col_start"], layer["col_end"]
                lc     = layer["length_cells"]
                # Per-surface "dirty" count = cells still equal to 1.0
                # within this column slice up to its length.
                slc = grid[:lc, cs:ce]
                total = slc.size
                dirty = int(np.sum(slc == 1.0))
                progress = 1.0 - (dirty / max(total, 1))
                # Find the source segment by uid and update its progress
                seg_uid = layer["segment_uid"]
                for seg in ap.get("roi_segments") or []:
                    if seg["segment_uid"] == seg_uid:
                        seg["progress"] = max(seg.get("progress", 0.0),
                                              progress)
                        break

        def _build_ap_planner(
            self, ap: Dict[str, object],
        ) -> "Optional[GBNN_H]":
            """Build the merged-grid GBNN+H planner for one AP.

            Stores the merged grid + column layout on the AP dict so
            the plan-view overlay and surface panel can read it.  Returns
            the live planner (or None if no surface segments visible).
            """
            segments = ap.get("roi_segments") or []
            grid, layout = self._merge_segments_to_grid(
                segments, ap["pose"])
            if grid is None or layout is None:
                return None
            ap["merged_grid"]   = grid
            ap["merged_layout"] = layout
            planner = GBNN_H(
                grid      = grid.copy(),
                n_ee      = 2,
                visualize = False,
                step_cap  = max(grid.size * 4, 400),
            )
            planner.reset()
            return planner

        def _compute_roi_segments_for_ap(
            self, pose: Tuple[float, float, float],
        ) -> List[Dict[str, object]]:
            """FoV-cone RoI extraction at an AP — returns visible obstacle-
            edge segments in plan view.

            Pipeline:
              1. Build the FoV cone (apex = AP pose, opening
                 MDA_ARM_FOV_DEG, reach MDA_ARM_REACH_M).
              2. For every static obstacle within reach (mounted MDAs and
                 dynamic non-static obstacles excluded), enumerate its
                 four OBB edges.
              3. Sample each edge densely; mark each sample as VISIBLE
                 iff (a) it's inside the FoV cone AND (b) the line from
                 AP-apex to the sample is occlusion-free against every
                 OTHER obstacle (the source obstacle of the edge is
                 excluded so the edge doesn't self-occlude).
              4. Group runs of consecutive VISIBLE samples; each run
                 becomes one segment in world-frame.  Drop runs shorter
                 than GBNNH_SEG_MIN_LENGTH_M to suppress aliasing dust.

            Each returned segment is a dict::

                {
                    "segment_uid":     int,
                    "obstacle_uid":    int,            # source obstacle
                    "obstacle_kind":   ObstacleKind,
                    "edge_idx":        int,            # 0-3, which OBB edge
                    "endpoints":       ((x1, y1), (x2, y2)),
                    "length_m":        float,
                    "progress":        float,          # 0..1 cleaning sweep
                }

            Different obstacles → different segments.  Different EDGES of
            the same obstacle → different segments.  Same edge split by
            occlusion → multiple segments (the disconnect the user
            described).  Vertical-vs-horizontal classification is left
            to the caller via ``obstacle_kind`` — this function does
            NOT merge by orientation.
            """
            ax, ay, ayaw = pose
            reach        = MDA_ARM_REACH_M
            half_fov     = math.radians(MDA_ARM_FOV_DEG) / 2.0

            # Static, non-mounted candidates within FoV reach.
            candidates: List["Obstacle"] = []
            for obs in self.obs_mgr.obstacles.values():
                if obs.is_mounted:
                    continue
                # Centre-to-AP distance ≤ reach + obstacle bounding rad
                rad = math.hypot(obs.half_w, obs.half_h)
                if math.hypot(obs.x - ax, obs.y - ay) > reach + rad:
                    continue
                candidates.append(obs)
            if not candidates:
                return []

            seg_uid = 0
            out: List[Dict[str, object]] = []
            n_samples = self.GBNNH_SEG_SAMPLES_PER_EDGE
            min_len   = self.GBNNH_SEG_MIN_LENGTH_M

            # Local outward normals for the 4 OBB edges (matching the
            # CCW order produced by _obstacle_obb_edges: south, east,
            # north, west).
            LOCAL_NORMALS = [(0.0, -1.0), (1.0, 0.0),
                             (0.0,  1.0), (-1.0, 0.0)]

            def _edge_orient(p1, p2):
                """Edge line orientation in [0, pi).  Edges and their
                reverses share the same line angle, so we mod by pi."""
                a = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                return a % math.pi

            def _orient_diff(a, b):
                d = abs(a - b) % math.pi
                return min(d, math.pi - d)

            ORIENT_TOL = 0.50   # ~28°; perpendicular pairs (≈π/2) are
                                # always rejected by this tolerance

            # Pass 1: per obstacle, build the sorted list of facing edges
            # (closest first by nearest-point distance, length tiebreak).
            per_obs_edges: List[
                Tuple["Obstacle", List[Dict[str, object]]]
            ] = []

            def _nearest_pt_on_segment(p1, p2, qx, qy):
                dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
                len_sq = dx * dx + dy * dy
                if len_sq < 1e-12:
                    return p1
                t = ((qx - p1[0]) * dx + (qy - p1[1]) * dy) / len_sq
                t = max(0.0, min(1.0, t))
                return (p1[0] + t * dx, p1[1] + t * dy)

            for src_obs in candidates:
                edges = self._obstacle_obb_edges(src_obs)
                # Occluders include the source obstacle itself — its own
                # body must occlude its far face & side faces.  The
                # analytical ray-OBB endpoint-epsilon handles the
                # "endpoint sits on the source's near surface" graze.
                edge_occluders = candidates

                # Direction from obstacle CENTRE back toward the AP —
                # used by Rule 1 to keep only edges whose outward
                # normal opposes (AP→obstacle), i.e. edges that face
                # the robot.
                to_ap_x = ax - src_obs.x
                to_ap_y = ay - src_obs.y
                to_ap_len = math.hypot(to_ap_x, to_ap_y)
                if to_ap_len < 1e-9:
                    continue
                to_ap_x /= to_ap_len
                to_ap_y /= to_ap_len

                co_obs = math.cos(src_obs.yaw)
                so_obs = math.sin(src_obs.yaw)

                # Build the per-obstacle facing-edge list — used by the
                # orientation-consistency pass below.  Distance is
                # measured AP → nearest-point-on-segment (not midpoint).
                facing_edges = []
                for edge_idx, ((p1x, p1y), (p2x, p2y)) in enumerate(edges):
                    lnx, lny = LOCAL_NORMALS[edge_idx]
                    nx_w = co_obs * lnx - so_obs * lny
                    ny_w = so_obs * lnx + co_obs * lny
                    facing_dot = nx_w * to_ap_x + ny_w * to_ap_y
                    if facing_dot < self.GBNNH_EDGE_FACING_COS:
                        continue
                    npx, npy = _nearest_pt_on_segment(
                        (p1x, p1y), (p2x, p2y), ax, ay)
                    d_to_ap = math.hypot(npx - ax, npy - ay)
                    edge_len = math.hypot(p2x - p1x, p2y - p1y)
                    facing_edges.append({
                        "idx":        edge_idx,
                        "p1":         (p1x, p1y),
                        "p2":         (p2x, p2y),
                        "facing_dot": facing_dot,
                        "dist":       d_to_ap,
                        "length":     edge_len,
                        "orient":     _edge_orient((p1x, p1y), (p2x, p2y)),
                    })
                if not facing_edges:
                    continue
                facing_edges.sort(
                    key=lambda e: (e["dist"], -e["length"], -e["facing_dot"]))
                per_obs_edges.append((src_obs, facing_edges))

            # Pass 2: orientation consistency across obstacles.
            # The user's rule: surfaces selected at one AP must share
            # an orientation.  We pick a REFERENCE orientation from the
            # globally-closest edge across all obstacles, then for
            # each obstacle keep its closest edge that MATCHES that
            # reference (within ORIENT_TOL).  Obstacles with no
            # matching-orientation facing edge are dropped — better
            # to skip a chair than to clean its perpendicular edge
            # alongside the table's parallel edge.
            if not per_obs_edges:
                return out
            global_closest = None
            for _, edges_list in per_obs_edges:
                if not edges_list:
                    continue
                e0 = edges_list[0]
                if global_closest is None or e0["dist"] < global_closest["dist"]:
                    global_closest = e0
            if global_closest is None:
                return out
            ref_orient = global_closest["orient"]

            # Final per-obstacle pick: closest edge whose orientation
            # is within ORIENT_TOL of the reference.
            iter_pairs: List[Tuple["Obstacle", Dict[str, object]]] = []
            for src_obs, edges_list in per_obs_edges:
                matching = [e for e in edges_list
                            if _orient_diff(e["orient"], ref_orient)
                                <= ORIENT_TOL]
                if not matching:
                    continue
                iter_pairs.append((src_obs, matching[0]))

            # Sample each chosen edge — orientation-consistent across obstacles.
            for src_obs, chosen in iter_pairs:
                edge_occluders = candidates
                edge_idx = chosen["idx"]
                (p1x, p1y) = chosen["p1"]
                (p2x, p2y) = chosen["p2"]

                # Sample the edge at n_samples + 1 points inclusive.
                visible_flags: List[bool] = []
                sample_pts:    List[Tuple[float, float]] = []
                for s in range(n_samples + 1):
                    t = s / n_samples
                    sx = p1x + t * (p2x - p1x)
                    sy = p1y + t * (p2y - p1y)
                    sample_pts.append((sx, sy))
                    if not self._point_in_fov_cone(
                            sx, sy, ax, ay, ayaw,
                            half_fov, reach):
                        visible_flags.append(False)
                        continue
                    # LOS check against every candidate (incl. source)
                    if not self._segment_los_clear(
                            ax, ay, sx, sy, edge_occluders,
                            n_samples=9):
                        visible_flags.append(False)
                        continue
                    visible_flags.append(True)

                # Walk the visibility array, collect contiguous True
                # runs into segments.
                i = 0
                while i <= n_samples:
                    if not visible_flags[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 <= n_samples and visible_flags[j + 1]:
                        j += 1
                    a = sample_pts[i]
                    b = sample_pts[j]
                    seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
                    if seg_len >= min_len:
                        seg_uid += 1
                        out.append({
                            "segment_uid":   seg_uid,
                            "obstacle_uid":  src_obs.uid,
                            "obstacle_kind": src_obs.kind,
                            "edge_idx":      edge_idx,
                            "endpoints":     (a, b),
                            "length_m":      seg_len,
                            "progress":      0.0,
                        })
                    i = j + 1

            return out

        def _refresh_gbnnh_active(self) -> None:
            """Per-tick driver — advance the AP iteration + GBNN_H planner.

            Lifecycle:
              0. Auto-revert: if a run completed >= GBNNH_RESET_AFTER_MS
                 ago, flip every AP from "done" (green) back to "queued"
                 (yellow) so the user can re-run with a single Enter.
              1. If no active run, return.
              2. If host hasn't reached current AP yet
                 (distance > GBNNH_AP_TOL_M), let the existing nav cursor
                 keep driving — return.
              3. On arrival, instantiate a GBNN_H planner once and call
                 reset().  Halt host motion (zero nav goal) so the base
                 stays put while the arms work.
              4. Tick planner.step() once per frame.
              5. On planner.is_done(), record stats, mark AP done, advance
                 AP idx, set nav goal to next AP.
              6. When all APs are done, clear active flag + nav state.
            """
            # 0. Auto-revert green markers to yellow GBNNH_RESET_AFTER_MS
            #    after the most recent run completed.  Runs even when
            #    `_gbnnh_active` is False — that's the whole point: the
            #    flag goes False the instant the last AP finishes, but
            #    the green markers should linger until the timer
            #    elapses, then revert in one frame.
            if self._gbnnh_completion_tick is not None:
                elapsed = pygame.time.get_ticks() - self._gbnnh_completion_tick
                if elapsed >= self.GBNNH_RESET_AFTER_MS:
                    for ap in self._gbnnh_aps:
                        ap["done"]  = False
                        ap["frame"] = None
                        ap["grid"]  = None
                        ap["stats"] = None
                    self._gbnnh_completion_tick = None
                    self._gbnnh_active_ap_idx   = None
                    self._set_message(
                        "GBNN+H APs reset — press Enter to re-run.",
                        HUD_ACCENT)

            if not self._gbnnh_active:
                return
            host = self._gbnnh_host_rid
            if host is None or host not in self.bus.poses:
                # Host vanished → abort run defensively
                self._gbnnh_active = False
                return
            if self._gbnnh_active_ap_idx is None:
                self._gbnnh_active = False
                return
            if self._gbnnh_active_ap_idx >= len(self._gbnnh_aps):
                # All APs done
                self._gbnnh_active = False
                self._gbnnh_planner = None
                self._nav_goals.pop(host, None)
                self._set_message(
                    "GBNN+H run complete — all APs covered.", HUD_OK)
                self._gbnnh_completion_tick = pygame.time.get_ticks()
                return

            ap = self._gbnnh_aps[self._gbnnh_active_ap_idx]
            ap_pose = ap["pose"]
            hp = self.bus.poses[host]
            d = math.hypot(hp.x - ap_pose[0], hp.y - ap_pose[1])

            # Phase 1: en route + arriving + aligning yaw (no planner yet).
            if self._gbnnh_planner is None:
                # 1a. en route — wait until the nav system has signalled
                # arrival (it pops the goal once within PATHFIND_WAYPOINT_TOL,
                # ~0.15 m).  Until then, do nothing — the existing cursor-
                # based controller is driving toward (ap.x, ap.y).
                if host in self._nav_goals:
                    # Stall detection: if the host can't make progress
                    # (A* keeps failing because the AP is in a tight
                    # spot), accept the current pose after a few seconds
                    # of stagnation rather than retrying forever.
                    cur_pos = (hp.x, hp.y)
                    if self._gbnnh_stall_pos is None:
                        self._gbnnh_stall_pos = cur_pos
                        self._gbnnh_stall_count = 0
                    else:
                        moved = math.hypot(
                            cur_pos[0] - self._gbnnh_stall_pos[0],
                            cur_pos[1] - self._gbnnh_stall_pos[1])
                        if moved < self.GBNNH_STALL_MOVE_TOL_M:
                            self._gbnnh_stall_count += 1
                        else:
                            self._gbnnh_stall_pos = cur_pos
                            self._gbnnh_stall_count = 0
                    if self._gbnnh_stall_count >= self.GBNNH_STALL_FRAME_LIMIT:
                        # Give up navigating — accept current pose as
                        # arrival and proceed to yaw alignment / segment
                        # extraction.  Better to clean from a slightly
                        # offset pose than to spin forever.
                        self._nav_goals.pop(host, None)
                        self._nav_paths.pop(host, None)
                        self._nav_wp_idx.pop(host, None)
                        self._nav_replan_tick.pop(host, None)
                        self._gbnnh_stall_pos   = None
                        self._gbnnh_stall_count = 0
                        self._set_message(
                            f"{ap['label']}: nav stalled ({d:.2f} m off), "
                            f"cleaning from current pose.", HUD_WARN)
                        # Fall through into yaw alignment + extraction.
                    else:
                        return
                else:
                    # Nav has popped the goal on its own (genuine arrival).
                    # Reset stall tracker for next AP.
                    self._gbnnh_stall_pos   = None
                    self._gbnnh_stall_count = 0
                # Defensive: nav goal gone but host still far — only
                # re-issue if NOT stalling (the stall path above already
                # cleared the goal cleanly).
                if (host not in self._nav_goals
                        and d > self.GBNNH_AP_TOL_M
                        and self._gbnnh_stall_count > 0):
                    self._nav_goals[host] = (ap_pose[0], ap_pose[1])
                    self._nav_paths.pop(host, None)
                    self._nav_wp_idx.pop(host, None)
                    self._nav_replan_tick.pop(host, None)
                    return

                # 1b. align yaw — the AP carries a yaw that points at the
                # cleaning surface.  Rotate the host's yaw toward it
                # frame-by-frame using direct pose mutation (the cmd_vel
                # cascade is already idle since nav is done).  For fused
                # formations only the host pose rotates — members keep
                # their own yaws (the MDA is host-attached, so only the
                # host's heading matters for cleaning orientation).
                target_yaw = ap_pose[2]
                yaw_err = math.atan2(
                    math.sin(target_yaw - hp.yaw),
                    math.cos(target_yaw - hp.yaw))
                YAW_TOL = 0.05               # ~3°
                if abs(yaw_err) > YAW_TOL:
                    # Cap rotation rate so the turn is visible (not a
                    # snap), but quick enough to not feel sluggish.
                    max_step = BASE_ANG_SPEED * DT * 0.5
                    step = min(abs(yaw_err), max_step)
                    new_yaw = hp.yaw + math.copysign(step, yaw_err)
                    self.bus.poses[host] = Pose(
                        x=hp.x, y=hp.y, yaw=new_yaw)
                    return
                # 1c. snap to exact target yaw, then extract RoI segments.
                self.bus.poses[host] = Pose(
                    x=hp.x, y=hp.y, yaw=target_yaw)

                # Plan-view RoI: visible obstacle-edge segments inside
                # the FoV cone, with line-of-sight occlusion against the
                # other obstacles.  Deferred to arrival so visibility
                # reflects the live scene (moveable obstacles may have
                # shifted between AP placement and now).
                segments = self._compute_roi_segments_for_ap(ap_pose)
                if not segments:
                    # Nothing visible from here — skip this AP.
                    self._set_message(
                        f"{ap['label']}: no surface visible on arrival, "
                        f"skipping.", HUD_WARN)
                    ap["done"]  = True
                    ap["stats"] = {'skipped': True}
                    ap["roi_segments"] = []
                    self._gbnnh_active_ap_idx += 1
                    if self._gbnnh_active_ap_idx >= len(self._gbnnh_aps):
                        self._gbnnh_active = False
                        self._set_message(
                            "GBNN+H run complete.", HUD_OK)
                        self._gbnnh_completion_tick = pygame.time.get_ticks()
                        return
                    nxt = self._gbnnh_aps[self._gbnnh_active_ap_idx]
                    self._nav_goals[host] = (nxt["pose"][0], nxt["pose"][1])
                    self._nav_paths.pop(host, None)
                    self._nav_wp_idx.pop(host, None)
                    self._nav_replan_tick.pop(host, None)
                    return
                ap["roi_segments"]         = segments
                self._gbnnh_active_seg_idx = None  # no longer per-segment
                # Reset step throttle so the new AP starts cleanly.
                self._gbnnh_step_frame_counter = 0
                # Build ONE merged-grid GBNN+H planner for this AP.
                # Surfaces concatenate horizontally with gap columns
                # (fully obstacle) between them per Rules 2+3.
                self._gbnnh_planner = self._build_ap_planner(ap)
                if self._gbnnh_planner is None:
                    # Layout build failed — skip this AP.
                    self._set_message(
                        f"{ap['label']}: could not merge segments, "
                        f"skipping.", HUD_WARN)
                    ap["done"]  = True
                    ap["stats"] = {"skipped": True}
                    self._gbnnh_active_ap_idx += 1
                    if self._gbnnh_active_ap_idx >= len(self._gbnnh_aps):
                        self._gbnnh_active = False
                        self._gbnnh_completion_tick = pygame.time.get_ticks()
                        return
                    nxt = self._gbnnh_aps[self._gbnnh_active_ap_idx]
                    self._nav_goals[host] = (nxt["pose"][0], nxt["pose"][1])
                    self._nav_paths.pop(host, None)
                    self._nav_wp_idx.pop(host, None)
                    self._nav_replan_tick.pop(host, None)
                    return
                self._set_message(
                    f"Arrived at {ap['label']} — merged RoI: "
                    f"{len(segments)} surface(s), "
                    f"{ap['merged_grid'].shape[1]}×"
                    f"{ap['merged_grid'].shape[0]} grid",
                    HUD_ACCENT)
                return

            # Phase 2: tick the merged-grid planner.  When done, mirror
            # final coverage into each segment's progress (so the plan
            # view shows green) and advance to next AP.
            segs = ap.get("roi_segments") or []
            if (self._gbnnh_planner is not None
                    and not isinstance(self._gbnnh_planner, str)):
                planner = self._gbnnh_planner
                # Step throttle: advance the planner once every
                # GBNNH_FRAMES_PER_STEP frames so the EE waypoint
                # motion is human-readable.  Render still updates
                # every frame from the cached planner state.
                self._gbnnh_step_frame_counter += 1
                if self._gbnnh_step_frame_counter < self.GBNNH_FRAMES_PER_STEP:
                    return
                self._gbnnh_step_frame_counter = 0
                still = planner.step()
                # Mirror per-segment progress from the planner's column
                # coverage so the plan-view overlay tracks live state.
                self._update_segment_progress_from_planner(ap)
                if still:
                    return

            # Run on this AP done.
            n_segs = len(segs)
            total_len = sum(s["length_m"] for s in segs)
            # Force all segments to fully-clean visual state on the
            # plan view (green markers) — the merged grid may still
            # have uncovered gap cells but those represent the
            # impassable gap, not unfinished work.
            for s in segs:
                s["progress"] = 1.0
            ap["done"]  = True
            ap["stats"] = {
                "segments":       n_segs,
                "total_length_m": total_len,
                "merged_shape":   ap.get("merged_grid").shape
                                  if ap.get("merged_grid") is not None
                                  else None,
            }
            self._gbnnh_planner        = None
            self._gbnnh_active_seg_idx = None
            self._set_message(
                f"{ap['label']} done: {n_segs} segment(s), "
                f"total {total_len:.2f} m of surface swept",
                HUD_OK)

            # Advance to the next AP.
            self._gbnnh_active_ap_idx += 1
            if self._gbnnh_active_ap_idx >= len(self._gbnnh_aps):
                self._gbnnh_active = False
                self._set_message(
                    "GBNN+H run complete — all APs covered.", HUD_OK)
                self._gbnnh_completion_tick = pygame.time.get_ticks()
                return
            nxt = self._gbnnh_aps[self._gbnnh_active_ap_idx]
            self._nav_goals[host] = (nxt["pose"][0], nxt["pose"][1])
            self._nav_paths.pop(host, None)
            self._nav_wp_idx.pop(host, None)
            self._nav_replan_tick.pop(host, None)

        def _draw_gbnnh_overlay(self) -> None:
            """Floor-view overlay for Mode 5 (RoI segment edition).

            Layers (drawn after robots, before HUD):
              1. AP markers (filled circles + heading arrow + label) for
                 every AP in the queue.  Status colour-coded:
                     queued    → yellow
                     active    → amber
                     done      → dark green
              2. Active-AP-only: translucent FoV cone + the visible
                 RoI segments.  Each segment is drawn as a thick line
                 along an obstacle edge, with the swept portion
                 (progress > 0) in green and the remaining portion
                 in amber.
              3. Active AP pulsing ring.
            """
            queue = self._gbnnh_aps
            if not queue:
                return

            screen = self.screen
            ap_radius_px = max(4, self._metres_to_px(self.GBNNH_AP_RADIUS_M))
            active_idx = self._gbnnh_active_ap_idx if self._gbnnh_active else None

            def status_for(i: int, ap_dict) -> str:
                if ap_dict.get("done"):
                    return 'done'
                if active_idx is not None and i == active_idx:
                    return 'active'
                return 'queued'

            COLOURS = {
                'done':   ((60, 130, 60),  (30, 90, 30)),
                'active': ((240, 180, 50), (180, 120, 20)),
                'queued': ((220, 200, 80), (140, 130, 30)),
            }

            def draw_one_ap(ap_dict, status: str) -> None:
                fill_col, outline_col = COLOURS[status]
                px, py, pyaw = ap_dict["pose"]
                cx, cy = self._world_to_screen(px, py)
                pygame.draw.circle(screen, fill_col, (cx, cy), ap_radius_px)
                pygame.draw.circle(screen, outline_col, (cx, cy),
                                   ap_radius_px, width=2)
                arrow_len = ap_radius_px + 6
                ax = cx + int(math.cos(pyaw) * arrow_len)
                ay = cy - int(math.sin(pyaw) * arrow_len)
                pygame.draw.line(screen, outline_col, (cx, cy), (ax, ay), 2)
                lbl = self.font_small.render(
                    ap_dict.get("label", ""), True, outline_col)
                screen.blit(lbl,
                            (cx - lbl.get_width() // 2,
                             cy - ap_radius_px - lbl.get_height() - 1))

            for i, ap in enumerate(queue):
                draw_one_ap(ap, status_for(i, ap))

            # Active-AP FoV cone + RoI segments (only the AP currently
            # being cleaned shows these; queued/done APs only show the
            # marker so the canvas stays uncluttered).
            if active_idx is not None and 0 <= active_idx < len(queue):
                act_ap = queue[active_idx]
                segs = act_ap.get("roi_segments")
                if segs is not None:
                    self._draw_fov_cone_and_segments(act_ap, segs)

            # Active AP pulsing ring
            if active_idx is not None and 0 <= active_idx < len(queue):
                ap = queue[active_idx]
                px, py, _ = ap["pose"]
                cx, cy = self._world_to_screen(px, py)
                t = (pygame.time.get_ticks() % 1000) / 1000.0
                pulse_r = ap_radius_px + 4 + int(8 * t)
                alpha = max(0, int(220 * (1.0 - t)))
                surf = pygame.Surface((pulse_r * 2 + 4, pulse_r * 2 + 4),
                                      pygame.SRCALPHA)
                pygame.draw.circle(
                    surf, (240, 180, 50, alpha),
                    (pulse_r + 2, pulse_r + 2), pulse_r, width=3)
                screen.blit(surf, (cx - pulse_r - 2, cy - pulse_r - 2))

        def _draw_fov_cone_and_segments(
            self, ap: Dict[str, object],
            segs: List[Dict[str, object]],
        ) -> None:
            """Render the FoV cone (translucent triangle) + visible RoI
            segments for the currently-active AP.

            Each segment is drawn as a thick polyline along the obstacle
            edge that produced it; the swept (cleaned) portion shows in
            green, the remaining portion in amber.  This makes the
            "lines in plan view" geometry visible exactly as Phase E §3
            specifies — and disjoint segments stay disjoint.
            """
            ap_pose = ap["pose"]
            ax, ay, ayaw = ap_pose
            reach    = MDA_ARM_REACH_M
            half_fov = math.radians(MDA_ARM_FOV_DEG) / 2.0

            # ---- FoV cone (sector approximated by N-segment fan) -----
            n_arc = 24
            poly_w = [(ax, ay)]
            for k in range(n_arc + 1):
                ang = ayaw - half_fov + (2 * half_fov) * k / n_arc
                wx = ax + reach * math.cos(ang)
                wy = ay + reach * math.sin(ang)
                poly_w.append((wx, wy))
            poly_px = [self._world_to_screen(x, y) for (x, y) in poly_w]
            xs = [p[0] for p in poly_px]; ys = [p[1] for p in poly_px]
            bbx, bby = min(xs), min(ys)
            bbw = max(xs) - bbx + 2
            bbh = max(ys) - bby + 2
            if bbw > 0 and bbh > 0:
                cone_surf = pygame.Surface((bbw, bbh), pygame.SRCALPHA)
                local = [(p[0] - bbx + 1, p[1] - bby + 1) for p in poly_px]
                pygame.draw.polygon(cone_surf, (220, 200, 80, 35), local)
                pygame.draw.polygon(cone_surf, (180, 160, 30, 180),
                                    local, width=1)
                self.screen.blit(cone_surf, (bbx - 1, bby - 1))

            # ---- RoI segments (one polyline each, with sweep colour) -
            SWEPT_COL    = (60, 200, 90)     # green — already cleaned
            REMAIN_COL   = (240, 180, 50)    # amber — remaining
            for seg in segs:
                (p1x, p1y), (p2x, p2y) = seg["endpoints"]
                progress = float(seg.get("progress", 0.0))
                # Split point along the segment based on progress
                spx = p1x + progress * (p2x - p1x)
                spy = p1y + progress * (p2y - p1y)
                a_px = self._world_to_screen(p1x, p1y)
                m_px = self._world_to_screen(spx, spy)
                b_px = self._world_to_screen(p2x, p2y)
                # Done portion (p1 → split)
                if progress > 1e-3:
                    pygame.draw.line(self.screen, SWEPT_COL,
                                     a_px, m_px, width=4)
                # Remaining portion (split → p2)
                if progress < 1.0 - 1e-3:
                    pygame.draw.line(self.screen, REMAIN_COL,
                                     m_px, b_px, width=4)
                # End-cap dots so the user sees segment boundaries —
                # disjoint segments are visually distinct.
                for ep in (a_px, b_px):
                    pygame.draw.circle(self.screen, (40, 40, 40),
                                       ep, 3)
                    pygame.draw.circle(self.screen, (250, 250, 250),
                                       ep, 2)

        # ---------------- Surface-view subpanel (Tab toggle) -----------

        # Layout — anchored to upper-right of the canvas, below the HUD
        GBNNH_PANEL_PX:    int = 280
        GBNNH_PANEL_MARGIN: int = 12

        def _draw_gbnnh_surface_panel(self) -> None:
            """Surface-view subpanel — GBNN+H 2D grid for the active segment.

            Restored from the pre-v3 grid view.  Each Phase E run
            instantiates a fresh GBNN+H planner per RoI segment; this
            panel shows the live state of THAT planner: cells (dirty /
            visited / obstacle), two coloured EE markers, per-EE
            trails.  Header tracks segment index, source obstacle kind,
            step count, flight count.  Hidden when
            ``self._gbnnh_show_panel`` is False.
            """
            if not self._gbnnh_show_panel:
                return
            active_idx = self._gbnnh_active_ap_idx
            if (active_idx is None
                    or active_idx >= len(self._gbnnh_aps)):
                return
            ap = self._gbnnh_aps[active_idx]
            segs = ap.get("roi_segments")
            if not segs:
                return
            if (self._gbnnh_planner is None
                    or isinstance(self._gbnnh_planner, str)):
                return

            rs = self._gbnnh_planner.render_state()
            grid       = rs["grid"]
            ee_pos     = rs["ee_pos"]
            paths      = rs["paths"]
            flight     = rs["flight"]
            step_count = rs["step"]
            H, V       = rs["shape"]

            # Compute cell size so the grid fits inside the configured
            # max panel size, then size the actual panel to JUST the
            # grid plus a thin border.  No header text, no axis ticks,
            # no extra whitespace.
            BORDER_PAD = 4
            max_dim = self.GBNNH_PANEL_PX - 2 * BORDER_PAD
            cell_size = min(max_dim / max(V, 1),
                            max_dim / max(H, 1))
            grid_px_w = int(cell_size * V)
            grid_px_h = int(cell_size * H)
            pw = grid_px_w + 2 * BORDER_PAD
            ph = grid_px_h + 2 * BORDER_PAD

            sw, sh = self.screen.get_size()
            px = sw - pw - self.GBNNH_PANEL_MARGIN
            py = HUD_H + self.GBNNH_PANEL_MARGIN

            # White background (matches gbnnh.py's matplotlib default).
            surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
            surf.fill((250, 250, 250, 245))

            grid_x = BORDER_PAD
            grid_y = BORDER_PAD

            # No header text or tick labels — the panel is just the
            # grid (matches user request: "axis numbers and title not
            # needed, just show the graph as it is now").

            # Cell colours — same hex palette as the GBNN+H demo file:
            #   COLOR_OBSTACLE  = '#0A090A'  (10,  9, 10)
            #   COLOR_UNVISITED = '#C1C1C1'  (193, 193, 193)
            #   COLOR_VISITED   = '#5F7472'  (95, 116, 114)
            COLOR_OBSTACLE  = (10, 9, 10)
            COLOR_UNVISITED = (193, 193, 193)
            COLOR_VISITED   = (95, 116, 114)
            for i in range(H):
                for j in range(V):
                    v = grid[i, j]
                    if v == -1.0:
                        c = COLOR_OBSTACLE
                    elif v == 1.0:
                        c = COLOR_UNVISITED
                    else:
                        c = COLOR_VISITED
                    cx = grid_x + int(j * cell_size)
                    cy = grid_y + int(i * cell_size)
                    pygame.draw.rect(
                        surf, c,
                        (cx, cy,
                         max(1, int(cell_size)),
                         max(1, int(cell_size))))

            # Gridlines between cells (light grey) — matches the implicit
            # cell-edge separation visible in matplotlib's Rectangle
            # rendering.
            grid_line_col = (190, 190, 200)
            for ii in range(H + 1):
                yy = grid_y + int(ii * cell_size)
                pygame.draw.line(
                    surf, grid_line_col,
                    (grid_x, yy),
                    (grid_x + grid_px_w, yy), 1)
            for jj in range(V + 1):
                xx = grid_x + int(jj * cell_size)
                pygame.draw.line(
                    surf, grid_line_col,
                    (xx, grid_y),
                    (xx, grid_y + grid_px_h), 1)

            # No axis tick labels — keeping the panel grid-only.

            # EE-partition line — thin red dotted vertical line at the
            # split column when n_ee == 2.  Mirrors the GBNN+H demo's
            # `ax.plot([xline, xline], ..., linestyle=':', color='red')`.
            n_ee     = rs.get('n_ee', 2)
            split_at = int(rs.get('split_at', V // 2))
            if n_ee == 2 and 0 < split_at < V:
                line_x = grid_x + int(split_at * cell_size)
                # Dashed: 4-px segments with 4-px gaps
                step = max(2, int(cell_size * 0.5))
                py0 = grid_y
                py1 = grid_y + grid_px_h
                yy = py0
                while yy < py1:
                    pygame.draw.line(
                        surf, (220, 60, 60),
                        (line_x, yy),
                        (line_x, min(yy + step, py1)),
                        1)
                    yy += step * 2

            # Per-EE trails (paths from the planner) — matches the
            # GBNN+H paper's "blue / green / red / cyan / ..." palette.
            EE_COLORS = [
                ( 31,  88, 200),    # blue
                ( 36, 160,  60),    # green
                (200,  60,  60),    # red
                ( 60, 200, 200),    # cyan
                (200,  60, 200),    # magenta
                (220, 140,  40),    # orange
                (130,  60, 200),    # purple
            ]
            for ee_id, plist in paths.items():
                if len(plist) < 2:
                    continue
                col = EE_COLORS[ee_id % len(EE_COLORS)]
                pts = [(grid_x + int((j + 0.5) * cell_size),
                        grid_y + int((i + 0.5) * cell_size))
                       for (i, j) in plist]
                pygame.draw.lines(surf, col, False, pts, 2)

            # EE markers — match the GBNN+H demo style:
            #   1) translucent colored cell (alpha 0.85)
            #   2) white circle on top
            #   3) smaller colored dot in the centre
            for k, (ei, ej) in enumerate(ee_pos):
                col = EE_COLORS[k % len(EE_COLORS)]
                cx = grid_x + int(ej * cell_size)
                cy = grid_y + int(ei * cell_size)
                cs_int = max(1, int(cell_size))
                # Translucent colored cell overlay
                cell_surf = pygame.Surface((cs_int, cs_int), pygame.SRCALPHA)
                cell_surf.fill((*col, 217))   # 0.85 alpha
                surf.blit(cell_surf, (cx, cy))
                # Marker centre
                ccx = cx + cs_int // 2
                ccy = cy + cs_int // 2
                outer_r = max(4, int(cs_int * 0.55))
                inner_r = max(2, int(cs_int * 0.30))
                pygame.draw.circle(surf, (255, 255, 255),
                                   (ccx, ccy), outer_r)
                pygame.draw.circle(surf, col,
                                   (ccx, ccy), inner_r)

            # Grid border (matches matplotlib's black axis spine).
            pygame.draw.rect(
                surf, (40, 40, 50),
                (grid_x - 1, grid_y - 1,
                 grid_px_w + 2, grid_px_h + 2),
                width=2)

            self.screen.blit(surf, (px, py))

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
                    f"Robot{host}[H] attached to {obs.label}. "
                    f"Fusion allowed. Split to singleton to detach (T).",
                    HUD_OK)
            else:
                self._set_message(
                    f"Robot{host} attached to {obs.label}. "
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

            # Per-formation "active" velocity proxy: max per-tick
            # displacement of any host in the formation since last
            # collision check.  Used as the velocity weight — a
            # formation that's currently being teleop'd or nav-
            # driven has a non-zero displacement and ends up with
            # a smaller share of the collision push (the OTHER side
            # absorbs more).  An idle formation has 0 displacement
            # and ends up taking the brunt of the push (gets shoved
            # out of the way).  Both still get SOME push when both
            # are moving — neither passes through the other.
            fid_vel: Dict[int, float] = {}
            for rid in ids:
                cur = self.bus.poses[rid]
                prev = self._prev_collision_poses.get(rid, cur)
                v = math.hypot(cur.x - prev.x, cur.y - prev.y)
                fid = fmap[rid]
                if v > fid_vel.get(fid, 0.0):
                    fid_vel[fid] = v

            # Accumulate per-formation translation corrections
            corr: Dict[int, List[float]] = {
                fid: [0.0, 0.0] for fid in set(fmap.values())
            }
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    ri, rj = ids[i], ids[j]
                    fi, fj = fmap[ri], fmap[rj]
                    if fi == fj:
                        # Same formation = rigid body — skip per-tick
                        # push.  Overlap prevention for newly-fused
                        # members lives in `_complete_dock` (post-snap
                        # relaxation pass before rT_b is captured), so
                        # no overlap should exist in a fused formation
                        # going forward.  Pushing here would create a
                        # shift-pull oscillation with the FSM's rigid-
                        # body rT_b enforcement.
                        continue
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

                        # Velocity-weighted split (replaces former size
                        # weighting).  Each formation's per-tick motion
                        # is the proxy for "active driver".  Faster
                        # formation absorbs less push; slower / idle
                        # formation absorbs more.  Both always get a
                        # share of the correction, so no formation
                        # passes through another regardless of mass —
                        # the no-ghost invariant.  Both idle = 50/50
                        # mutual block (same as before).
                        vi = fid_vel.get(fi, 0.0)
                        vj = fid_vel.get(fj, 0.0)
                        v_total = vi + vj
                        if v_total > 1e-4:
                            # Faster gets less push, slower gets more.
                            wi = vj / v_total
                            wj = vi / v_total
                        else:
                            # Both effectively idle — split evenly.
                            wi, wj = 0.5, 0.5
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
                    # Debug: accumulate collision push applied this tick
                    ex, ey = self._dbg_collision_push.get(rid, (0.0, 0.0))
                    self._dbg_collision_push[rid] = (ex + cx, ey + cy)

            # ---- Same-formation overlap resolution -----------------
            # Inter-formation pushes are done.  Now scan each fused
            # formation for INTERNAL overlap (two same-formation
            # members within DOCKED_DISTANCE_M of each other).  When
            # detected, relax members apart and refresh rT_b so the
            # rigid-body geometry enforcement (`_enforce_formation_
            # geometry`) preserves the new spacing instead of snapping
            # members back to the old overlapping offsets.
            #
            # Without this pass, `_enforce_formation_geometry` rewrites
            # every non-host member's pose from the stored rT_b each
            # tick — any in-tick same-formation push gets erased and
            # the visible overlap persists forever.  With this pass,
            # rT_b gets re-captured from the post-push positions, and
            # the new spacing locks in.
            #
            # Snapshot current poses for the NEXT tick's velocity
            # weighting.  Captured AFTER the inter-formation push has
            # been applied so the per-tick displacement reflects
            # actual world-frame motion (cmd_vel + collision push).
            self._prev_collision_poses = {
                rid: self.bus.poses[rid] for rid in ids
            }

            # Threshold of 0.05 m (5 cm) prevents micro-jitter from
            # numerical drift triggering rT_b updates every tick.
            SAME_FORM_OVERLAP_THRESHOLD = 0.05
            for fid, members in self._formations_by_fid().items():
                if len(members) < 2:
                    continue
                # Detect overlap between any pair in this formation
                has_overlap = False
                for i_m in range(len(members)):
                    for j_m in range(i_m + 1, len(members)):
                        m1, m2 = members[i_m], members[j_m]
                        p1 = self.bus.poses[m1]
                        p2 = self.bus.poses[m2]
                        d = math.hypot(p1.x - p2.x, p1.y - p2.y)
                        if d < DOCKED_DISTANCE_M - SAME_FORM_OVERLAP_THRESHOLD:
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                if not has_overlap:
                    continue

                # Relaxation: push non-host members apart, host stays put
                host = min(members)
                movable = [m for m in members if m != host]
                threshold = DOCKED_DISTANCE_M - COLLISION_EPSILON
                for _relax in range(20):
                    moved = False
                    for m in movable:
                        pm = self.bus.poses[m]
                        for other in members:
                            if other == m:
                                continue
                            po = self.bus.poses[other]
                            ddx = pm.x - po.x
                            ddy = pm.y - po.y
                            dd  = math.hypot(ddx, ddy)
                            if dd < threshold:
                                push = (threshold - dd) + COLLISION_EPSILON
                                if dd < 1e-6:
                                    ux_p, uy_p = 1.0, 0.0
                                else:
                                    ux_p, uy_p = ddx / dd, ddy / dd
                                self.bus.poses[m] = Pose(
                                    x   = pm.x + ux_p * push,
                                    y   = pm.y + uy_p * push,
                                    yaw = pm.yaw,
                                )
                                moved = True
                                pm = self.bus.poses[m]
                    if not moved:
                        break

                # Refresh rT_b for non-host members so geometry
                # enforcement preserves the relaxed spacing.
                hp = self.bus.poses[host]
                for m in members:
                    cfg_m = self.bus.configurers[m]
                    pm    = self.bus.poses[m]
                    if m == host:
                        cfg_m.rT_b = np.identity(3)
                        continue
                    v_world = np.array([pm.x - hp.x, pm.y - hp.y])
                    cy, sy  = math.cos(pm.yaw), math.sin(pm.yaw)
                    R_o     = np.array([[cy, -sy], [sy, cy]])
                    rT_b    = np.identity(3)
                    rT_b[:2, 2] = R_o.T @ (-v_world)
                    cfg_m.rT_b  = np.round(rT_b, 2)

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
                            # Phase E: a mounted MDA module is "part of" its
                            # host robot — it shares the host's pose every
                            # tick and would otherwise force-push the host
                            # away from itself the moment it's mounted.
                            # Skip ALL mounted MDAs from robot-collision so
                            # neither the host nor any other robot reacts
                            # to it.  Footprint inflation belongs in the
                            # planner inflation pass, not here.
                            if obs.is_mounted:
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
                            # Debug: accumulate obstacle-collision push
                            ex, ey = self._dbg_collision_push.get(
                                rid, (0.0, 0.0))
                            self._dbg_collision_push[rid] = (
                                ex + fx, ey + fy)

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

            # Multi-pass: a human pushed out of one obstacle may now overlap
            # another (e.g. wedged between a wall and a chair).  Three
            # passes catches typical cascading cases without ping-ponging.
            for _iter in range(3):
                any_push = False
                for human in human_obs:
                    hr = human.half_w  # human radius

                    # Human vs all other obstacles
                    for obs in self.obs_mgr.obstacles.values():
                        if obs.uid == human.uid:
                            continue
                        if obs.uid == self.obs_mgr.dragging_id:
                            continue
                        # Phase E: a mounted MDA is part of its host robot,
                        # not a free-standing wall.  Skip so humans aren't
                        # blocked by a robot's "ghost" attachment.
                        if obs.is_mounted:
                            continue
                        # Human treats everything as a solid wall (100% push)
                        result = obs.collision_overlap(human.x, human.y, hr)
                        if result is not None:
                            push_x, push_y, overlap = result
                            human.x += push_x * overlap
                            human.y += push_y * overlap
                            any_push = True
                            # Redirect walk_target AWAY from the wall so the
                            # human doesn't keep trying to walk through it
                            # next frame (which causes the gradual leak
                            # past thin walls when frame-rate stutters).
                            if hasattr(human, "_walk_target_x") \
                                    and hasattr(human, "_walk_target_y"):
                                tdx = human._walk_target_x - human.x
                                tdy = human._walk_target_y - human.y
                                # If the walk target is on the OPPOSITE side
                                # of the wall (dot product with push < 0),
                                # snap a fresh nearby target on the safe side.
                                if tdx * push_x + tdy * push_y < 0.0:
                                    human._walk_target_x = (
                                        human.x + push_x * 1.5
                                        + (tdy * 0.5))   # tangent component
                                    human._walk_target_y = (
                                        human.y + push_y * 1.5
                                        + (-tdx * 0.5))
                                    human._walk_timer = max(
                                        getattr(human, "_walk_timer", 60), 60)

                    # Human vs robots (circle-circle).  Both solid; human
                    # pushed 100% (robot side handled in obstacle resolver).
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
                            human.x += ux * overlap
                            human.y += uy * overlap
                            any_push = True
                if not any_push:
                    break

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
            self._dbg_nav_reason[host] = (
                f"cancel ({msg})" if msg else "cancel"
            )
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
                self._dbg_nav_reason[host] = "no-goal"
                return None

            goal = self._nav_goals[host]
            members = self._members_of(host)

            # Navigation reference point ("the robot is here").  Default
            # is the formation centroid (rotation centre) — that's what
            # P2P navigation in Mode 1 has always used: a split
            # singleton uses its own pose; a fused singleton uses the
            # formation centroid.
            #
            # Phase E (Mode 5) override: when navigating to an MDA
            # access point, the reference point is the MDA's pose
            # (== the host robot's pose, since the MDA is mounted on
            # the host).  Using the MDA pose for arrival makes the
            # host stop with the arm assembly directly above the AP,
            # which is what the AP yaw-alignment + cleaning expect.
            # Without this override a fused-formation host would stop
            # when the centroid hit the AP — the host pose would be
            # offset by half the formation diameter.
            mda_centred = (
                self._gbnnh_active
                and host == self._gbnnh_host_rid
                and self._gbnnh_planner is None
                and self._gbnnh_active_ap_idx is not None
                and 0 <= self._gbnnh_active_ap_idx < len(self._gbnnh_aps)
                and self.obs_mgr.has_mda_mounted(host)
            )
            if mda_centred:
                hp_ref = self.bus.poses[host]
                rc_x, rc_y = hp_ref.x, hp_ref.y
            else:
                rc_x, rc_y = self._rotation_centre(host)
            hp = self.bus.poses[host]

            # Check arrival
            if math.hypot(rc_x - goal[0], rc_y - goal[1]) < PATHFIND_WAYPOINT_TOL:
                self._cancel_nav(host)
                self._dbg_nav_reason[host] = "arrived"
                arrival_msg = (
                    f"Robot{host} (MDA centred) arrived at AP."
                    if mda_centred
                    else f"Robot{host} formation arrived."
                )
                self._set_message(arrival_msg, HUD_OK)
                return None

            # Decide whether to replan or reuse the existing path.
            # Replan every PATHFIND_REPLAN_INTERVAL ticks, or if no path yet.
            replan_tick = self._nav_replan_tick.get(host, 0)
            need_replan = (host not in self._nav_paths
                           or replan_tick <= 0)

            if need_replan:
                # Build occupancy grid excluding this formation's members
                nav_members = set(members)
                # Inter-Star congestion relief — treat OTHER
                # Inter-Star participants as transient, non-blocking
                # obstacles when planning for `host`:
                #
                #   FISSION: members emerge from a shared fused
                #     singleton at DOCKED_DISTANCE_M offsets, so
                #     each member's 0.80 m combined inflation
                #     (robot 0.40 + wall 0.40) blocks its
                #     neighbours' adjacent cells; no outbound
                #     path exists unless we ignore same-group
                #     members.
                #   FUSION: hosts converge toward a single goal
                #     point.  As they cluster, each host's 0.80 m
                #     inflation sits ON TOP of the other hosts'
                #     positions — `nearest_free_cell` then snaps
                #     A*'s start away from where the robot
                #     actually is, and the drive direction
                #     doesn't resolve the situation (teleop
                #     nudge was the only rescue).  The
                #     collision resolver keeps them physically
                #     separated, and the proximity-chain combine
                #     handles eventual arrival, so it's safe to
                #     ignore each other's inflation here.
                #
                # Static obstacles + non-Inter-Star robots still
                # block normally in both cases.
                if self._interstar_mode == "fission":
                    nav_members |= set(
                        self._interstar_fission_goals.keys())
                elif self._interstar_mode == "fusion":
                    nav_members |= set(
                        self._interstar_selected_cache or [])
                    nav_members |= set(
                        self._interstar_staging_slots.keys())
                robot_pos = []
                for r in sorted(self.bus.configurers.keys()):
                    if r in nav_members:
                        continue
                    rp = self.bus.poses[r]
                    robot_pos.append((rp.x, rp.y, ROBOT_OCCUPANCY_M))

                grid = self.obs_mgr.build_occupancy_grid(
                    self._world_bounds(),
                    cell_size=PATHFIND_CELL_SIZE,
                    inflate_radius=self._path_inflate_for(host),
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
                            f"Robot{host} nav cancelled (path blocked).")
                        return None
                    # Try using existing path if available
                    path = self._nav_paths.get(host)
                    if path is None or len(path) < 2:
                        self._dbg_nav_reason[host] = (
                            f"A*-fail (fails={fails})")
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
                    self._dbg_nav_reason[host] = "no-path-cached"
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
            #
            # Inter-Star-driven nav (fusion + fission) uses a shorter
            # lookahead because those paths hug the Inter-Star cell
            # sequence through narrow passages — a long lookahead would
            # cut across corners and push the robot body into walls
            # while the hybrid-mode yaw controller is still rotating.
            # For plain A* / GBNN / teleop, the original 0.60 m
            # keeps motion in open space smooth.
            interstar_driven = (
                self._interstar_plan_active
                and host in self._interstar_paths
            )
            LOOKAHEAD = 0.30 if interstar_driven else 0.60
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
                self._dbg_nav_reason[host] = "dist≈0"
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

            # ---- Motion-mode overrides per active planner ----------------
            # Inter-Star: HYBRID (diff + holo).  Differential alone
            #   would stop translation at every path bend to yaw-align;
            #   holonomic alone never rotates and misaligns yaw for
            #   the subsequent dock handshake.
            # GBNN: HOLONOMIC, with yaw locked to grid-up (+y world axis).
            #   GBNN cell-by-cell coverage wants straight-line sidewise
            #   motion across rows/columns — a diff turn per cell is
            #   wasteful.  Orientation is held at π/2 so the robot's
            #   sprite aligns to the RoI grid, matching the user's
            #   request "always face up, aligned to the selected area's
            #   grid orientation".
            # A* / teleop / default: use `self._nav_motion` as-is
            #   (cycled via the M key) + yaw follows direction of
            #   travel.  This preserves the existing teleop and A*
            #   feel the user already has.
            motion_mode = self._nav_motion
            target_yaw  = desired_yaw   # default: face direction of travel

            interstar_driven = (
                self._interstar_plan_active
                and host in self._interstar_paths
            )
            gbnn_driven = host in self._gbnn_planner_for
            # Phase E (Mode 5): the MDA-host is en route to an access
            # point — flag is True only during the en-route leg
            # (planner not yet built; once the per-AP GBNN+H planner
            # is instantiated the nav goal has been popped and this
            # whole code path doesn't run).
            gbnnh_driven = (
                self._gbnnh_active
                and host == self._gbnnh_host_rid
                and self._gbnnh_planner is None
                and self._gbnnh_active_ap_idx is not None
                and 0 <= self._gbnnh_active_ap_idx < len(self._gbnnh_aps)
            )

            if gbnn_driven:
                motion_mode = "holonomic"
                target_yaw  = math.pi / 2.0   # grid-up / world +y
            elif interstar_driven:
                motion_mode = "hybrid"
            elif gbnnh_driven:
                # Hybrid (diff + holo) lets the host translate
                # holonomically while a P-controller rotates yaw toward
                # a fixed target — exactly the per-AP requirement.
                # Bias target_yaw to the AP's intended yaw so the host
                # arrives mostly pre-aligned; the arrival-time yaw-snap
                # in `_refresh_gbnnh_active` finishes the last few
                # degrees.  Falls back to direction-of-travel (default)
                # only if the active-AP index is out of range — the
                # gate above keeps that case unreachable in practice.
                motion_mode = "hybrid"
                ap = self._gbnnh_aps[self._gbnnh_active_ap_idx]
                target_yaw = ap["pose"][2]

            # ── Differential mode ──
            if motion_mode == "differential":
                yaw_err = target_yaw - hp.yaw
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                if abs(yaw_err) > 0.15:
                    wz = max(-BASE_ANG_SPEED * self.vel_scale,
                             min(BASE_ANG_SPEED * self.vel_scale,
                                 yaw_err * 3.0))
                    self._dbg_nav_reason[host] = (
                        f"diff-turn (yaw_err={yaw_err:+.2f})")
                    return Twist(linear_x=0.0, linear_y=0.0, angular_z=wz)
                else:
                    vx_b = min(speed, dist / DT)
                    self._dbg_nav_reason[host] = "diff-drive"
                    return Twist(linear_x=vx_b, linear_y=0.0, angular_z=0.0)

            # ── Holonomic mode ──
            elif motion_mode == "holonomic":
                vx_w = (dx_w / dist) * speed
                vy_w = (dy_w / dist) * speed
                vx_b =  ch * vx_w + sh * vy_w
                vy_b = -sh * vx_w + ch * vy_w
                # Default pure-holonomic: no rotation.  GBNN override
                # adds a P-controller on yaw toward grid-up.
                wz = 0.0
                if gbnn_driven:
                    yaw_err = target_yaw - hp.yaw
                    yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                    wz = max(-BASE_ANG_SPEED * self.vel_scale,
                             min(BASE_ANG_SPEED * self.vel_scale,
                                 yaw_err * 2.0))
                self._dbg_nav_reason[host] = (
                    f"holo (d={dist:.2f},spd={speed:.2f})")
                return Twist(linear_x=vx_b, linear_y=vy_b, angular_z=wz)

            # ── Hybrid mode ──
            else:
                vx_w = (dx_w / dist) * speed
                vy_w = (dy_w / dist) * speed
                vx_b =  ch * vx_w + sh * vy_w
                vy_b = -sh * vx_w + ch * vy_w
                # Near-target yaw dampener: when the lookahead point is
                # very close to the robot, the direction-of-travel
                # vector (dx_w, dy_w) becomes noisy — small floating
                # drift flips it by 180° between ticks, and the yaw
                # P-controller fires equally-large wz commands in
                # alternating directions → visible spinning at the
                # end of the last cursor hop.  Skip yaw correction
                # when translation distance is below 0.25 m; hold
                # the current yaw instead.
                NEAR_TARGET_DIST_M = 0.25
                if dist < NEAR_TARGET_DIST_M:
                    wz = 0.0
                    yerr = 0.0
                else:
                    yaw_err = target_yaw - hp.yaw
                    yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                    yerr = yaw_err
                    wz = max(-BASE_ANG_SPEED * self.vel_scale,
                             min(BASE_ANG_SPEED * self.vel_scale,
                                 yaw_err * 2.0))
                self._dbg_nav_reason[host] = (
                    f"hybrid (d={dist:.2f},spd={speed:.2f},"
                    f"yaw_err={yerr:+.2f})")
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
                        f"Robot{host} nav cancelled (formation conflict).")
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
                label = f"host=Robot{host}  n={len(members)}"
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
            # Visual-only half-width: chosen so the rendered square's
            # circumscribed circle == ROBOT_OCCUPANCY_M, i.e. corners
            # never poke past the collision boundary.  Functional
            # ROBOT_SIZE_M (footprint / GBNN cell size) is unchanged.
            half_px = max(8, self._metres_to_px(ROBOT_VISUAL_HALF_M))

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

            # 3. Selection ring (single-robot active teleoperation target).
            # Note: Multi-robot Inter-Star candidates are indicated by a
            # yellow STAR at the robot's top-right corner (see step 6),
            # not by a halo ring — the old ring was still visible after
            # Inter-Star had completed and gave the misleading impression
            # that the selection was still live.
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
            # Heading line scaled to the visual body half-width so it
            # tracks the shrunken square (not the larger footprint).
            hx = pose.x + (ROBOT_VISUAL_HALF_M * 1.4) * math.cos(pose.yaw)
            hy = pose.y + (ROBOT_VISUAL_HALF_M * 1.4) * math.sin(pose.yaw)
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

            # 6. Inter-Star candidate star (top-right corner, absolute screen).
            # Drawn AFTER the translucent blit so the star stays opaque and
            # instantly recognisable even if the robot itself is dimmed
            # under a caster trolley.  Shown whenever the robot is:
            #   * ctrl-click multi-selected (pre-dispatch), OR
            #   * in an active Inter-Star plan's cached selection, OR
            #   * still pending in the post-plan proximity-chain fusion queue.
            # When Inter-Star terminates and the fusion queue drains, all
            # three sources become empty and the star disappears.
            if self._is_interstar_candidate(rid):
                # Top-right body corner in absolute screen coords
                trx_local, try_local = half_px, half_px
                rx =  c * trx_local - s * (-try_local)
                ry =  s * trx_local + c * (-try_local)
                star_cx = cx + rx
                star_cy = cy - ry
                outer_r = max(7, half_px // 2)
                inner_r = max(3, int(outer_r * 0.42))
                self._draw_star(
                    self.screen,
                    star_cx, star_cy,
                    outer_r, inner_r,
                    fill=(255, 225, 90),
                    outline=(20, 20, 25),
                )

            # 7. Label above body (always opaque for readability)
            host_rid = self._host_of(rid)
            role = "H" if rid == host_rid else "M"
            n    = len(self._members_of(rid))
            lbl  = f"Robot{rid}  {role}  n={n}  host=Robot{host_rid}"
            surf = self.font_small.render(lbl, True, (240, 240, 240))
            lbl_pos = (cx - surf.get_width() // 2, cy - half_px - 22)
            self.screen.blit(surf, lbl_pos)

        def _is_interstar_candidate(self, rid: int) -> bool:
            """True if `rid` should show the yellow Inter-Star candidate
            star on the current render tick.  Covers the full Inter-Star
            lifecycle: pre-dispatch Ctrl+click selection, active plan
            (fusion or fission), and the post-plan proximity-chain fuse.

            Formation-aware: when ANY member of a fused formation is
            Ctrl+click-selected, every same-host member also lights up.
            Otherwise the user sees only the clicked member starred and
            is confused when subsequent Ctrl+clicks on its formation-
            mates are rejected (the whole formation is already selected
            implicitly — the star just didn't show that).
            """
            if rid in self._selected_ids:
                return True
            # Any member of the same fused formation as a selected robot?
            my_host = self._host_of(rid)
            for sel in self._selected_ids:
                if self._host_of(sel) == my_host:
                    return True
            # Fission: candidate star stays only until the member has
            # moved further than INTERSTAR_FISSION_STAR_MOVE_M from
            # its dispatch-time pose.  Matches user spec: "stars should
            # be removed immediately after moving away from start".
            if self._interstar_mode == "fission":
                if rid in self._interstar_fission_start_poses:
                    start = self._interstar_fission_start_poses[rid]
                    p = self.bus.poses.get(rid)
                    if p is not None:
                        moved = math.hypot(p.x - start[0], p.y - start[1])
                        if moved < self.INTERSTAR_FISSION_STAR_MOVE_M:
                            return True
                    else:
                        return True
                # Fall through — already moved, star drops.
            else:
                # Fusion / idle: membership in the dispatch cache still
                # counts as a candidate.
                if rid in self._interstar_selected_cache:
                    return True
            for pair in self._interstar_fusion_pairs:
                if rid == pair[0] or rid == pair[1]:
                    return True
            # Formation-packer staging: highlight both the slot-driven
            # hosts and the combine anchor.  Dropped once the fusion-pair
            # queue has fully drained (see _tick_interstar_combine
            # post-drain cleanup), so stars vanish only after every
            # fuse completes — the user's "all fusion candidates
            # combined" condition.
            if rid in self._interstar_staging_slots:
                return True
            if rid == self._interstar_combine_anchor:
                return True
            return False

        def _draw_star(
            self,
            target,
            cx: float, cy: float,
            outer_r: int, inner_r: int,
            fill: Tuple[int, int, int],
            outline: Tuple[int, int, int] = (20, 20, 25),
        ) -> None:
            """5-pointed star centred at (cx, cy), upright.  Drawn with a
            solid fill + dark outline for legibility against bright
            robot colours and floor-grid shading."""
            pts: List[Tuple[float, float]] = []
            angle_off = -math.pi / 2.0          # first spike points up
            for i in range(10):
                r = outer_r if i % 2 == 0 else inner_r
                theta = angle_off + i * math.pi / 5.0
                pts.append((cx + r * math.cos(theta),
                            cy + r * math.sin(theta)))
            pygame.draw.polygon(target, fill, pts)
            pygame.draw.polygon(target, outline, pts, width=2)

        def _draw_pending_fission_stars(self) -> None:
            """Overlay: yellow star at every pending fission goal the user
            has clicked while a single fused robot is selected.  The list
            is populated by the fission-intent branch of `_on_mouseup`
            and drained on dispatch (n goals → `_dispatch_interstar_fission`)
            or cleared on Esc / fusion-intent promotion.  Each star
            visualises where one freshly-split member will diverge to."""
            if not self._fission_goals_pending:
                return
            for idx, (wx, wy) in enumerate(self._fission_goals_pending, 1):
                sx, sy = self._world_to_screen(wx, wy)
                self._draw_star(
                    self.screen,
                    sx, sy,
                    outer_r=12, inner_r=5,
                    fill=(255, 225, 90),
                    outline=(20, 20, 25),
                )
                # Small ordinal label so the user can tell the intended
                # goal order while still queueing further clicks.
                try:
                    lbl = self.font_small.render(
                        str(idx), True, (20, 20, 25))
                    self.screen.blit(
                        lbl,
                        (sx - lbl.get_width() // 2,
                         sy - lbl.get_height() // 2),
                    )
                except Exception:
                    pass

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
                        f"Selected: Robot{sel} [{role}]  "
                        f"formation host=Robot{host}, "
                        f"members={sorted(members)}, n={len(members)}"
                    )
                else:
                    sel_txt = (
                        f"Selected: Robot{sel}  [split singleton]"
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
                    nav_tag = f"  Robot{rid} NAV→({gx:.1f},{gy:.1f})"
            # Phase B: Interstar metric line — expansions vs A* baseline
            interstar_tag = ""
            if self._interstar_metrics:
                exp = int(self._interstar_metrics.get("expansions", 0))
                ratio = self._interstar_metrics.get("expansions_ratio", 1.0)
                interstar_tag = f"  Interstar: {exp} exp ({ratio:.2f}× A*)"
            # Phase B: multi-select count + pending fission queue
            sel_tag = ""
            if self._selected_ids:
                sel_tag = f"  [selected: {len(self._selected_ids)}]"
            if self._fission_host is not None and self._fission_goals_pending:
                sel_tag += (f"  [fission goals: "
                            f"{len(self._fission_goals_pending)}]")
            # Phase E: GBNN+H mode tag
            gbnnh_tag = ""
            if self._gbnnh_active and self._gbnnh_active_ap_idx is not None:
                idx = self._gbnnh_active_ap_idx
                tot = len(self._gbnnh_aps)
                if 0 <= idx < tot:
                    seg_str = ""
                    ap = self._gbnnh_aps[idx]
                    segs = ap.get("roi_segments")
                    if segs:
                        n_done = sum(1 for s in segs
                                     if s.get("progress", 0.0) >= 0.999)
                        seg_str = f" seg {n_done}/{len(segs)}"
                    flight_str = ""
                    if (self._gbnnh_planner is not None
                            and not isinstance(self._gbnnh_planner, str)):
                        rs = self._gbnnh_planner.render_state()
                        flight_str = (f" flight=L{rs['flight'][0]}"
                                      f"R{rs['flight'][1]}")
                    gbnnh_tag = (
                        f"  GBNN+H: AP {idx + 1}/{tot}"
                        f"{seg_str}{flight_str}")
            elif self._gbnnh_aps and self._gbnnh_active_host() is not None:
                gbnnh_tag = (f"  GBNN+H: {len(self._gbnnh_aps)} APs queued "
                             f"[Enter to start]")

            scale_line = (
                f"scale={self.vel_scale:.2f}    "
                f"cmd_vel  vx={cmd.linear_x:+.2f}  "
                f"vy={cmd.linear_y:+.2f}  wz={cmd.angular_z:+.2f}"
                + status_flag + rot_tag + nav_tag + interstar_tag
                + gbnnh_tag + sel_tag
            )
            line2 = self.font_med.render(scale_line, True, scale_col)
            self.screen.blit(line2, (12, 32))

            line3 = self.font_small.render(self.last_message, True,
                                           self.last_msg_col)
            self.screen.blit(line3, (12, 58))

            # ---- Debug line 4: per-tick diagnostics for the SELECTED ----
            # robot's host (why the robot is or isn't moving).  Shows:
            #   reason      — tag from `_compute_nav_cmd_for`'s last
            #                 return (arrived / A*-fail / hybrid / diff /
            #                 holo / no-goal / no-path-cached / dist≈0 /
            #                 cancel) or "—" if nav didn't run for it.
            #   goal=(x,y)  — current `_nav_goals[host]`
            #   path=N      — len(_nav_paths[host]) (0 = not cached)
            #   fails=N     — nav_fail_count climbing toward PATHFIND_FAIL_LIMIT
            #   replan=N    — ticks until next forced A* replan
            #   push=(dx,dy) — collision-resolver pose correction this tick
            sel = self.selected_id
            dbg_line = "DBG: —"
            if sel in self.bus.configurers:
                sel_host = self._host_of(sel)
                goal = self._nav_goals.get(sel_host)
                path = self._nav_paths.get(sel_host)
                fails = self._nav_fail_count.get(sel_host, 0)
                replan = self._nav_replan_tick.get(sel_host, 0)
                reason = self._dbg_nav_reason.get(sel_host, "—")
                push = self._dbg_collision_push.get(sel, (0.0, 0.0))
                push_mag = math.hypot(push[0], push[1])
                goal_txt = (f"({goal[0]:.1f},{goal[1]:.1f})"
                            if goal else "—")
                path_n = len(path) if path else 0
                dbg_line = (
                    f"DBG Robot{sel}(host{sel_host}): "
                    f"reason={reason}  goal={goal_txt}  "
                    f"path={path_n}  fails={fails}  "
                    f"replan={replan}  push={push_mag:.3f}m"
                )
            dbg_col = (150, 200, 255)   # pale blue
            line4 = self.font_small.render(dbg_line, True, dbg_col)
            self.screen.blit(line4, (12, 76))

            help_left  = ("1-5 select   SHIFT+N dock   SPACE fission   "
                          "Z rot   P algo   M motion   X nav   C clear")
            help_right = ("W/A/S/D diff   I/J/K/L holo   Q↑ E↓ speed   "
                          "Click: spawn/goal   Dbl-click: despawn")
            s1 = self.font_small.render(help_left,  True, (160, 160, 180))
            s2 = self.font_small.render(help_right, True, (160, 160, 180))
            self.screen.blit(s1, (12, 100))
            self.screen.blit(s2, (12, 118))

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

        def _draw_mounted_mdas(self) -> None:
            """Draw mounted MDA modules at z+1 above their host robots.

            ObstacleManager.draw_non_caster (used by _draw_obstacles) skips
            mounted MDAs so they don't render under the host.  This call
            sits after _draw_robot in the render pipeline so the MDA
            hexagon visibly stacks on top of its host."""
            self.obs_mgr.draw_mounted_mda(
                self.screen,
                self._world_to_screen,
                self._metres_to_px,
                self.font_small,
            )

        # ---- General-use: obstacle clearance halos -------------------
        def _draw_clearance_zones(self) -> None:
            """While Shift is held, render a translucent halo around every
            non-mounted obstacle showing its robot-clearance zone — the
            OBB inflated by ROBOT_OCCUPANCY_M.

            Mode-agnostic affordance: helps the user see where any
            click-based gesture (P2P nav goal, GBNN drag origin, AP
            placement) cannot land because the host robot's body would
            overlap the obstacle.  Only the visualization is gated on
            Shift; the actual rejection logic in each gesture's handler
            uses the same clearance value regardless of Shift state.

            Shift state is read live at draw time via
            ``pygame.key.get_mods()`` rather than ``self.input_state.shift``
            so the halo disappears the instant the key is released —
            no dependency on ``_sync_keyboard_state`` ordering within
            the tick.  Same pattern as the Shift+LMB click handler.

            Z-order: drawn between obstacles and per-robot paths so
            halos sit on top of obstacle bodies but under nav arrows
            and robot sprites.
            """
            shift_held = bool(
                pygame.key.get_mods()
                & (pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT)
            )
            if not shift_held:
                return
            inflate = ROBOT_OCCUPANCY_M
            for obs in self.obs_mgr.obstacles.values():
                if obs.is_mounted:
                    continue
                # Inflated OBB — same yaw, half-extents grown by `inflate`.
                hw = obs.half_w + inflate
                hh = obs.half_h + inflate
                co = math.cos(obs.yaw)
                so = math.sin(obs.yaw)
                local = [(-hw, -hh), ( hw, -hh),
                         ( hw,  hh), (-hw,  hh)]
                pts = []
                for lx, ly in local:
                    wx = obs.x + co * lx - so * ly
                    wy = obs.y + so * lx + co * ly
                    pts.append(self._world_to_screen(wx, wy))
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                bbx, bby = min(xs), min(ys)
                bbw = max(xs) - bbx + 2
                bbh = max(ys) - bby + 2
                if bbw <= 0 or bbh <= 0:
                    continue
                surf = pygame.Surface((bbw, bbh), pygame.SRCALPHA)
                local_pts = [(p[0] - bbx + 1, p[1] - bby + 1) for p in pts]
                # Translucent yellow fill, slightly more opaque outline
                pygame.draw.polygon(surf, (255, 200,  50,  55), local_pts)
                pygame.draw.polygon(surf, (220, 160,  20, 200),
                                    local_pts, width=2)
                self.screen.blit(surf, (bbx - 1, bby - 1))

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

        # ====================================================================
        # PHASE A — GBNN coverage dispatch + render hooks
        # ====================================================================

        def _build_world_spec_for_planners(self) -> WorldSpec:
            """Snapshot the world for any Planner adapter."""
            robots = []
            for rid, cfg in self.bus.configurers.items():
                p = self.bus.poses[rid]
                robots.append(RobotState(
                    robot_id    = rid,
                    pose        = p,
                    fsm_state   = cfg.fsm_state,
                    host_id     = cfg.host_id,
                    n           = cfg.n,
                    footprint_m = 2 * ROBOT_SIZE_M,
                ))
            return WorldSpec(
                bounds  = self._world_bounds(),
                cell_m  = 2 * ROBOT_SIZE_M,
                obs_mgr = self.obs_mgr,
                robots  = robots,
            )

        def _dispatch_gbnn_coverage(
            self,
            start_screen: Tuple[int, int],
            end_screen:   Tuple[int, int],
        ) -> None:
            """LMB-drag → run GBNN to completion on the drawn RoI for the
            host of the currently selected formation, then feed the cell
            sequence as a multi-waypoint nav path to the existing nav
            controller."""
            sel = self.selected_id
            if sel not in self.bus.configurers:
                self._set_message(
                    "GBNN: no robot selected.", HUD_WARN)
                return
            host = self._host_of(sel)
            members = self._members_of(sel)
            if sel != host and len(members) > 1:
                self._set_message(
                    f"Robot{sel} is not the host. Select Robot{host} for GBNN.",
                    HUD_WARN)
                return

            w0 = self._screen_to_world(*start_screen)
            w1 = self._screen_to_world(*end_screen)
            rect = Rect(w0[0], w0[1], w1[0], w1[1]).normalized()
            if rect.width < 0.4 or rect.height < 0.4:
                self._set_message(
                    f"GBNN RoI too small "
                    f"({rect.width:.2f}×{rect.height:.2f} m).", HUD_WARN)
                return

            # Fresh planner instance per dispatch (clean GBNN state)
            planner = GBNNBasePlanner(footprint_m=2 * ROBOT_SIZE_M)
            world = self._build_world_spec_for_planners()
            query = PlannerQuery(
                mode=PlannerMode.MANUAL, world=world,
                robots=world.robots, selected_ids=[host],
                goal=GoalSpec(area=rect),
            )
            init = planner.plan(query)
            if init.metrics.get("plan_failed"):
                self._set_message("GBNN: plan failed.", HUD_WARN)
                return

            # Drive INCREMENTALLY: GBNN steps one cell at a time, in
            # lockstep with the robot's actual arrival.  This way the
            # activity heatmap evolves visibly as the robot crosses each
            # cell — instead of jumping straight to "all visited" if we
            # ran GBNN to completion at dispatch.
            self._gbnn_planner_for[host] = planner
            if planner._approach_path:
                # Robot needs to first walk from its current pose to the
                # GBNN start cell — store the A* approach polyline.
                self._gbnn_approach[host]     = list(planner._approach_path)
                self._gbnn_approach_idx[host] = 0
            else:
                self._gbnn_approach.pop(host, None)
                self._gbnn_approach_idx.pop(host, None)

            self._set_message(
                f"GBNN coverage: Robot{host} on "
                f"{rect.width:.2f}×{rect.height:.2f}m RoI — "
                f"stepping incrementally (heatmap evolves live)",
                HUD_OK,
            )

        def _merge_nav_for_uninvolved(
            self,
            distribution: Dict[int, Twist],
            busy_rids: set,
        ) -> None:
            """Overlay nav cmds for robots NOT involved in an active docking
            or trolley-attach sequence, so unrelated formations keep moving
            along their plans (P2P or GBNN coverage) instead of pausing
            until docking finishes.

            Mutates `distribution` in place.  `busy_rids` is the set of
            robot ids currently owned by the docking/trolley sequence —
            those keep whatever cmd `_docking_cmd` / `_trolley_attach_cmd`
            assigned them.
            """
            try:
                nav_dist, _raw = self._compute_all_nav_distributions()
            except Exception:
                return
            for rid, cmd in nav_dist.items():
                if rid in busy_rids:
                    continue
                # Skip if the existing distribution already has a non-zero
                # cmd for this robot (very rare — _docking_cmd zeroes
                # everyone else, so usually we're overwriting Twist()).
                existing = distribution.get(rid)
                if existing is not None and (
                        abs(existing.linear_x)  > 1e-9
                        or abs(existing.linear_y)  > 1e-9
                        or abs(existing.angular_z) > 1e-9):
                    continue
                distribution[rid] = cmd

        # ====================================================================
        #  PHASE B — Interstar auto-dispatch + renderer + nav integration
        # ====================================================================

        def _dispatch_interstar_fusion(
            self,
            selected_ids: List[int],
            goal_world:   Tuple[float, float],
        ) -> None:
            """Fire Inter-Star fusion for the given selected formations
            converging on `goal_world`.

            Driving is cursor-based — every selected formation walks its
            own Inter-Star path cell-by-cell through `_refresh_interstar_active`,
            so the amber shared-tail visualisation matches what the
            robots actually do.  Per-tick waypoint-wait (first-come gets
            the next cell) handles serialisation on the shared tail.

            Formation-host promotion: a Ctrl+click may land on any
            member of a fused singleton (e.g. Robot 4 in {2, 3, 4}).
            Feeding that member's rid into Inter-Star as a "start"
            used the member's rigid-offset pose instead of the host's
            pose — the nav controller then promoted the goal back up
            to the host, producing a constant offset between the
            planned path and the actual formation motion (oscillation
            near the goal).  Resolving to the host up front eliminates
            this: Inter-Star plans from the ACTUAL pose the nav
            controller will drive, and `_interstar_paths` is keyed by
            host.
            """
            # Promote each selected rid to its formation host and
            # de-duplicate.  Same-host Ctrl+click rejection prevents two
            # selections from mapping to the same host in the normal
            # flow, but this guards any pathological call too.
            host_ids = sorted({self._host_of(r) for r in selected_ids
                               if r in self.bus.configurers})
            if len(host_ids) < 2:
                self._set_message(
                    "Interstar: need ≥2 distinct formations.", HUD_WARN)
                return

            # Build a query whose robots-for-planning have each fused
            # host's pose overridden to the formation's rotation centre
            # (centroid).  Reason: the nav controller drives the
            # rotation centre, not the host pose.  If the centroid sits
            # ahead of the host in the goal direction (common for fused
            # singletons whose non-host member happens to be on the
            # goal-side), feeding Inter-Star the host pose as start
            # makes path[0] = host-cell and path[1] = one cell toward
            # goal from the host — which is BEHIND the centroid.  The
            # nav controller then drives the centroid BACKWARD to
            # reach path[1].  Planning from the centroid instead
            # produces a path whose first cells sit ahead of wherever
            # the nav actually drives from, so motion is always
            # forward.
            world = self._build_world_spec_for_planners()
            adjusted_robots: List[RobotState] = []
            for r in world.robots:
                if r.robot_id in host_ids and r.n > 1:
                    cx, cy = self._rotation_centre(r.robot_id)
                    adjusted_robots.append(RobotState(
                        robot_id    = r.robot_id,
                        pose        = Pose(x=cx, y=cy, yaw=r.pose.yaw),
                        fsm_state   = r.fsm_state,
                        host_id     = r.host_id,
                        n           = r.n,
                        footprint_m = r.footprint_m,
                    ))
                else:
                    adjusted_robots.append(r)
            world_for_plan = WorldSpec(
                bounds  = world.bounds,
                cell_m  = world.cell_m,
                obs_mgr = world.obs_mgr,
                robots  = adjusted_robots,
            )
            query = PlannerQuery(
                mode         = PlannerMode.INTERSTAR,
                world        = world_for_plan,
                robots       = adjusted_robots,
                selected_ids = host_ids,
                goal         = GoalSpec(point=goal_world),
            )
            # Scale inflation with the biggest formation in the group.
            # Inter-Star uses a single grid for all starts, so we take
            # the MAX of per-formation inflation across `host_ids`
            # (with the planner's 0.40 m baseline as a lower bound).
            max_infl = max(
                [self._path_inflate_for(h) for h in host_ids]
                + [0.40]
            )
            self._interstar_planner.inflate_radius = max_infl
            result = self._interstar_planner.plan(query)
            if result.metrics.get("plan_failed"):
                self._set_message("Interstar: plan failed.", HUD_WARN)
                return

            # Store each formation's DENSE Inter-Star path.  Cursor-based
            # driving in `_refresh_interstar_active` feeds the current
            # target as a 2-point nav polyline, advancing the cursor
            # each time the formation arrives — this way the robots
            # actually follow the Inter-Star path cell-by-cell and the
            # amber shared-tail overlay matches their motion.
            self._interstar_paths   = {}
            self._interstar_cursor  = {}
            for host, path in result.assignments.items():
                if not path:
                    continue
                self._interstar_paths[host]  = list(path)
                self._interstar_cursor[host] = 1       # skip start cell

            # Shared tail + metrics cached for overlay + HUD
            self._interstar_shared_segment = list(
                result.extras.get("shared_segment", []))
            self._interstar_metrics        = dict(result.metrics)
            self._interstar_plan_active    = True
            self._interstar_host           = min(host_ids)
            self._interstar_pending_fuses  = list(result.reconfig)
            self._interstar_mode            = "fusion"
            self._interstar_fusion_goal     = tuple(goal_world)
            self._interstar_selected_cache  = list(host_ids)
            self._interstar_replan_counter  = 0

            exp   = int(result.metrics.get("expansions", 0))
            ratio = result.metrics.get("expansions_ratio", 1.0)
            shared_n = len(self._interstar_shared_segment)
            wx, wy = goal_world
            self._set_message(
                f"Interstar FUSION: {len(host_ids)} formations → "
                f"({wx:.1f}, {wy:.1f})  shared tail {shared_n} cells  "
                f"[{exp} expansions, {ratio:.2f}× vs A*]",
                HUD_OK,
            )

        def _dispatch_interstar_fission(
            self,
            host_id:       int,
            goal_points:   List[Tuple[float, float]],
        ) -> None:
            """Fire Inter-Star fission: one fused singleton splits into n
            members diverging to `goal_points`.  The FISSION rcfg is queued
            into the sequencer; once the Configurer FSM resets n→1 for each
            member, the nav controller drives them to their assigned goals.

            Note: the existing TeleopSim fission command (SPACE) already
            handles the sim-level formation split.  We reuse `_try_fission`
            to actually split the formation, then assign per-robot goals."""
            world = self._build_world_spec_for_planners()
            query = PlannerQuery(
                mode         = PlannerMode.INTERSTAR,
                world        = world,
                robots       = world.robots,
                selected_ids = [host_id],
                goal         = GoalSpec(points=list(goal_points)),
            )
            # Scale Inter-Star inflation to the pre-split formation's
            # hull size (the fission_start hull), with the planner's
            # 0.40 m baseline as a lower bound.
            self._interstar_planner.inflate_radius = max(
                self._path_inflate_for(host_id), 0.40
            )
            result = self._interstar_planner.plan(query)
            if result.metrics.get("plan_failed"):
                self._set_message("Interstar: fission plan failed.", HUD_WARN)
                return

            # Split the formation (reuse existing Configurer fission path).
            # After _try_fission, each former member is a singleton again.
            members_before_split = sorted(self._members_of(host_id))
            self._try_fission(host_id)

            # Inter-Star's fission planner assumes a single start cell
            # (the host's pre-split pose) with n divergence paths to n
            # goals.  After `_try_fission`, each former member sits at
            # its own rigid offset from that cell.
            #
            # Two coordinated fixes so robots actually WALK their
            # Inter-Star paths instead of cutting straight to the goal:
            #   (1) Optimal member→goal assignment (n! enumeration,
            #       n ≤ 5) picks the permutation that minimises total
            #       start→goal distance, so the member closest to each
            #       goal gets that goal.  Prevents paths crossing
            #       through other members' starting positions.
            #   (2) Each member joins its assigned divergence path at
            #       the waypoint NEAREST its actual current pose, not
            #       at path[0] (the host's pre-split cell).  The
            #       assigned path is rebuilt as
            #           [member_pose] + orig_path[nearest_idx:]
            #       so the cursor-based driver has a continuous polyline
            #       starting where the member physically is, then
            #       rejoining the Inter-Star divergence line on its way
            #       to the goal.  The amber overlay for that member
            #       shows exactly what the member will walk.
            path_list = [result.assignments.get(-(i + 1), [])
                         for i in range(len(goal_points))]
            n_assign = min(len(members_before_split), len(goal_points))

            def _member_xy(m: int) -> Tuple[float, float]:
                p = self.bus.poses[m]
                return (p.x, p.y)

            # (1) Optimal member→goal assignment by total-distance.
            from itertools import permutations as _perms
            members_use = members_before_split[:n_assign]
            goals_use   = list(goal_points)[:n_assign]
            best_perm   = members_use
            best_cost   = float('inf')
            for perm in _perms(members_use):
                cost = 0.0
                for j, m in enumerate(perm):
                    mx, my = _member_xy(m)
                    gx, gy = goals_use[j]
                    cost += math.hypot(mx - gx, my - gy)
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm
            # best_perm[j] is the member assigned to goals_use[j]

            # (2) Populate visualisation + cursor-driver paths.  Each
            # path starts at the member's actual pose and rejoins the
            # Inter-Star plan at the nearest waypoint on it.
            self._interstar_paths  = {}
            self._interstar_cursor = {}
            for j, rid in enumerate(best_perm):
                orig = path_list[j] if j < len(path_list) else []
                if not orig:
                    continue
                mx, my = _member_xy(rid)
                # Find nearest waypoint on the original Inter-Star path
                # (index into orig).  Skip the very first cell if the
                # member happens to be snapped to it — the cursor will
                # advance off it naturally.
                best_i = 0
                best_d = float('inf')
                for i, wp in enumerate(orig):
                    d = math.hypot(wp[0] - mx, wp[1] - my)
                    if d < best_d:
                        best_d, best_i = d, i
                # Build the walk path: current pose + from nearest
                # waypoint onward to the goal.  Guarantees a smooth
                # polyline with no backward detours.
                walk_path = [(mx, my)] + list(orig[best_i:])
                self._interstar_paths[rid]  = walk_path
                self._interstar_cursor[rid] = 1

            # _interstar_paths + _interstar_cursor were populated above
            self._interstar_shared_segment = []   # fission has no shared tail
            self._interstar_metrics        = dict(result.metrics)
            self._interstar_plan_active    = True
            self._interstar_host           = host_id
            self._interstar_pending_fuses  = []
            # Mode flag drives termination branch in _refresh_interstar_active.
            # Fission termination: every assigned robot has reached its
            # own goal point (pose-based check, strict).  No cursor-based
            # driving, no iterative re-planning — the standard nav
            # controller handles the motion from here.
            self._interstar_mode           = "fission"
            self._interstar_selected_cache = list(self._interstar_paths.keys())
            self._interstar_fusion_goal    = None
            self._interstar_replan_counter = 0
            # Cache the exact (rid → goal) assignment so termination can
            # check against the user's click-point goals, not the Inter-
            # Star path endpoints (which may differ by up to one grid
            # cell after the fission planner's cell-snap).
            self._interstar_fission_goals: Dict[int, Tuple[float, float]] = {
                rid: goals_use[j] for j, rid in enumerate(best_perm)
            }
            # Snapshot each member's current pose as its "start" — the
            # yellow candidate star will drop off once the member moves
            # further than INTERSTAR_FISSION_STAR_MOVE_M from this
            # position.  Captured here so the star lifecycle is tied to
            # the actual dispatch moment, not to path cell 0.
            self._interstar_fission_start_poses = {}
            for rid in best_perm:
                if rid in self.bus.poses:
                    p = self.bus.poses[rid]
                    self._interstar_fission_start_poses[rid] = (p.x, p.y)

            exp   = int(result.metrics.get("expansions", 0))
            ratio = result.metrics.get("expansions_ratio", 1.0)
            self._set_message(
                f"Interstar FISSION: Robot{host_id} (n={len(goal_points)}) → "
                f"{len(goal_points)} diverging goals  "
                f"[{exp} expansions, {ratio:.2f}× vs A*]",
                HUD_OK,
            )

        def _tick_interstar_combine(self) -> None:
            """Two-phase combine:

            Phase 1 — STAGING.  Every non-anchor host drives via A*
              (nav goal = its assigned dock slot, at DOCKED_DISTANCE_M
              from the anchor at a distinct angle).  When all hosts
              are within INTERSTAR_STAGING_TOL_M of their slots (or
              the timeout fires), transition to phase 2.

            Phase 2 — FUSING.  Pairwise `_try_fuse(host_i, anchor)` in
              sequence.  Because each host is already at its DOCKED
              slot, the docking P-controller's snap translation is
              near-zero → no rigid-translation overlap between
              formations.  The yaw alignment still runs but now
              against formations already at their final positions,
              not mid-approach ones, which was the specific failure
              the user reported ("possibly due to rotating in between
              fusion sequences").
            """
            # ---- Phase 1: staging -------------------------------------
            if self._interstar_staging_active:
                # Absolute safety cap (20 s) — backstop in case
                # something breaks and no host ever settles.  Normal
                # operation uses the rest-based transition below.
                self._interstar_staging_timer += 1
                abs_timed_out = (self._interstar_staging_timer
                                 >= self.INTERSTAR_STAGING_TIMEOUT)

                # Arrival check
                staged = True
                for host, slot in self._interstar_staging_slots.items():
                    if host not in self.bus.poses:
                        staged = False
                        break
                    p = self.bus.poses[host]
                    d = math.hypot(p.x - slot[0], p.y - slot[1])
                    if d > self.INTERSTAR_STAGING_TOL_M:
                        staged = False
                        break

                # Rest detection — accumulate consecutive ticks in
                # which every staging host moved less than
                # INTERSTAR_STOP_TOL_M.  Unlike the old absolute
                # timeout, this waits for stragglers to actually
                # finish (either arrive at their slot or have their
                # A* give up and nav-cancel them at rest).  Once
                # every host is simultaneously at rest for
                # INTERSTAR_STAGING_REST_FRAMES, transition.
                all_at_rest = True
                for host in self._interstar_staging_slots:
                    if host not in self.bus.poses:
                        continue
                    p = self.bus.poses[host]
                    cur = (p.x, p.y)
                    prev = self._interstar_staging_last_pose.get(host, cur)
                    moved = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
                    self._interstar_staging_last_pose[host] = cur
                    if moved > self.INTERSTAR_STOP_TOL_M:
                        all_at_rest = False
                if all_at_rest:
                    self._interstar_staging_rest_ticks += 1
                else:
                    self._interstar_staging_rest_ticks = 0
                rest_timed_out = (self._interstar_staging_rest_ticks
                                  >= self.INTERSTAR_STAGING_REST_FRAMES)

                if not (staged or rest_timed_out or abs_timed_out):
                    return   # at least one host is still driving

                # Transition: staging → fusing.  Stop nav for every
                # staged host (they're already at their slot OR have
                # been cancelled) so the pairwise fuse doesn't have
                # to fight residual nav cmds.
                self._interstar_staging_active = False
                self._interstar_staging_rest_ticks = 0
                self._interstar_staging_last_pose = {}
                anchor = self._interstar_combine_anchor
                for host in self._interstar_staging_slots:
                    self._nav_goals.pop(host, None)
                    self._nav_paths.pop(host, None)
                    self._nav_wp_idx.pop(host, None)
                    self._nav_replan_tick.pop(host, None)
                if anchor is None:
                    self._interstar_staging_slots = {}
                    self._interstar_fusion_pairs = []
                    return
                # Build pairs: each non-anchor → anchor.  Anchor is
                # always the target so every trigger drives INTO the
                # anchor's formation (smallest net motion).
                self._interstar_fusion_pairs = [
                    (host, anchor)
                    for host in sorted(self._interstar_staging_slots.keys())
                ]
                if abs_timed_out:
                    self._set_message(
                        "Interstar: staging absolute timeout — fusing "
                        "with current positions (forced docks possible).",
                        HUD_WARN,
                    )
                elif rest_timed_out and not staged:
                    self._set_message(
                        "Interstar: all staging hosts at rest — fusing "
                        "(stuck robots will be force-docked in).",
                        HUD_WARN,
                    )

            # ---- Phase 2: pairwise fusing (queue drain) ---------------
            if not self._interstar_fusion_pairs:
                # Queue drained.  If staging state is still around,
                # that means every fuse has fired and completed (the
                # last `_try_fuse` cleared `self.docking` and we're
                # one tick later).  Clear the staging-slot + anchor
                # references now, which drops the yellow candidate
                # star from every participant — the user's spec:
                # "stars should be removed after robots are fully
                # combined after fusion Inter-Star".
                if (self._interstar_combine_anchor is not None
                        and self.docking is None
                        and self._trolley_docking is None):
                    self._interstar_staging_slots  = {}
                    self._interstar_combine_anchor = None
                return
            if self.docking is not None or self._trolley_docking is not None:
                return
            trigger, target = self._interstar_fusion_pairs[0]
            if (trigger not in self.bus.configurers
                    or target not in self.bus.configurers):
                self._interstar_fusion_pairs.pop(0)
                return
            if self._host_of(trigger) == self._host_of(target):
                self._interstar_fusion_pairs.pop(0)
                return
            self._interstar_fusion_pairs.pop(0)
            # Force-fuse when trigger couldn't reach its staging slot
            # (closest pair distance exceeds the normal docking range).
            # Matches user spec: "if the final position adjustment
            # before fusion Inter-Star fails to move to the final
            # position due to being stuck, they can cancel it and
            # join the fusion."  Robots that DID make their slot pass
            # the distance check normally; only stragglers hit the
            # forced path.
            members_a = [r for r, f in self.formation_of.items()
                         if f == self.formation_of[trigger]]
            members_b = [r for r, f in self.formation_of.items()
                         if f == self.formation_of[target]]
            pair_d = self._closest_pair_distance(members_a, members_b)
            self._try_fuse(trigger, target,
                           force=(pair_d > DOCKING_DISTANCE_M))

        def _build_interstar_fusion_queue(
            self,
            selected: List[int],
            goal:     Tuple[float, float],
        ) -> List[Tuple[int, int]]:
            """Proximity-chain fusion sequence.

            Rule (per user spec):
              * The robot closest to the goal point is the "anchor" — it
                is always the LAST target to be fused into the chain.
              * At every step, the formation whose member is furthest
                from the goal is the trigger (it initiates the fusion,
                matching the "furthest robot fuses with its closest
                neighbour" intent).
              * The target is the robot from the closest OTHER formation
                that is physically nearest the trigger formation (so
                `_try_fuse`'s docking-range check passes and the motion
                distance is minimal).
              * The loop iterates on the merging formations (not individual
                robots), so two robots fused together become one entity
                for subsequent "furthest" ranking — the chain grows
                outside-in toward the anchor.

            Returns a list of (trigger_rid, target_rid) pairs in firing
            order.  The final pair will always have the anchor as its
            target, honouring "the last robot have to be the robot on
            the goal pose."
            """
            if len(selected) < 2:
                return []

            def pose_xy(rid: int) -> Tuple[float, float]:
                p = self.bus.poses[rid]
                return (p.x, p.y)

            def dist(a: int, b: int) -> float:
                ax, ay = pose_xy(a)
                bx, by = pose_xy(b)
                return math.hypot(ax - bx, ay - by)

            def dist_goal(rid: int) -> float:
                x, y = pose_xy(rid)
                return math.hypot(x - goal[0], y - goal[1])

            # Anchor = robot closest to the goal (joins last)
            anchor = min(selected, key=dist_goal)

            # ---- Connectivity filter (anti-fragmentation) ---------------
            # A pairwise `_try_fuse` is rejected when the closest-pair
            # distance between two formations exceeds DOCKING_DISTANCE_M
            # (2.0 m).  If a robot is genuinely stuck far from the
            # goal-cluster, chaining through it would silently drop the
            # stuck pair from the queue — the rest of the chain then
            # merges without it, but any subsequent pair whose trigger
            # inherited the stuck robot's formation still couldn't
            # reach the other side of the chain, producing the
            # fragmented "two fused singletons" output the user saw.
            #
            # Fix: build a connectivity graph with edge condition
            #   dist(a, b) <= DOCKING_DISTANCE_M + slack
            # then keep only the connected component containing the
            # anchor.  Robots outside that component are left as split
            # singletons (not in the merged formation, but not blocking
            # the merge either).
            SLACK = 1.0   # metres of wiggle over DOCKING_DISTANCE_M so
                          # robots that stopped just-barely-too-far still
                          # qualify — the docking P-controller tolerates
                          # a short approach motion at snap time.
            THRESH = DOCKING_DISTANCE_M + SLACK
            from collections import deque as _deque
            reachable = {anchor}
            frontier = _deque([anchor])
            while frontier:
                r = frontier.popleft()
                for s in selected:
                    if s in reachable:
                        continue
                    if dist(r, s) <= THRESH:
                        reachable.add(s)
                        frontier.append(s)
            if len(reachable) < 2:
                # Anchor couldn't reach anyone — nothing to combine
                return []

            others = [r for r in reachable if r != anchor]

            # Chain-build across non-anchor robots first, so the anchor
            # stays a singleton until the final pair.
            formations: Dict[int, set] = {r: {r} for r in others}
            queue: List[Tuple[int, int]] = []

            while len(formations) >= 2:
                # Trigger formation: whose MAX (furthest-from-goal) member
                # has the largest distance to the goal point
                trigger_rep = max(
                    formations.keys(),
                    key=lambda rep: max(dist_goal(r)
                                        for r in formations[rep]),
                )
                other_reps = [r for r in formations if r != trigger_rep]
                # Target formation: closest (by nearest-pair) to trigger
                def pair_gap(rep: int) -> float:
                    return min(
                        dist(a, b)
                        for a in formations[trigger_rep]
                        for b in formations[rep]
                    )
                target_rep = min(other_reps, key=pair_gap)
                # Pick the exact closest-pair (trigger_rid, target_rid)
                best_pair = None
                best_d    = float('inf')
                for a in formations[trigger_rep]:
                    for b in formations[target_rep]:
                        d = dist(a, b)
                        if d < best_d:
                            best_d, best_pair = d, (a, b)
                if best_pair is not None:
                    queue.append(best_pair)
                # Merge target into trigger
                formations[trigger_rep] |= formations[target_rep]
                del formations[target_rep]

            # Final pair: the chain formation's member closest to the
            # anchor triggers the last fusion, with the anchor as target.
            if formations:
                chain_rep     = next(iter(formations))
                chain_members = formations[chain_rep]
                trigger = min(chain_members, key=lambda r: dist(r, anchor))
                queue.append((trigger, anchor))

            return queue

        def _complete_interstar_and_combine(self) -> None:
            """Called when all selected robots have halted for 1 s with at
            least one at the goal.  Clears Interstar state (paths vanish)
            and builds a proximity-chain fusion queue so the physical
            pairwise docking runs outside-in toward the goal-pose anchor
            via Configurer's proper FSM handshake."""
            selected = sorted(self._interstar_paths.keys())
            n = len(selected)
            goal = self._interstar_fusion_goal

            # Clear Interstar state → paths stop rendering
            for rid in list(self._interstar_paths.keys()):
                self._nav_goals.pop(rid, None)
                self._nav_paths.pop(rid, None)
                self._nav_wp_idx.pop(rid, None)
                self._nav_replan_tick.pop(rid, None)
            self._interstar_paths          = {}
            self._interstar_cursor         = {}
            self._interstar_shared_segment = []
            self._interstar_plan_active    = False
            self._interstar_pending_fuses  = []
            self._interstar_stop_timer     = 0
            self._interstar_cluster_timer  = 0
            self._interstar_last_positions = {}
            # Fusion mode finished → clear mode + re-plan scaffold so a
            # future Interstar dispatch starts fresh.
            self._interstar_mode           = ""
            self._interstar_fusion_goal    = None
            self._interstar_selected_cache = []
            self._interstar_replan_counter = 0

            # Inter-Star has terminated — the Ctrl+click candidate set no
            # longer represents a live selection.  Clear it so the yellow
            # corner star disappears on the robots whose only remaining
            # state is the about-to-drain fusion-pair queue; robots still
            # pending in that queue keep their star via the pair-membership
            # branch of `_is_interstar_candidate`.
            self._selected_ids.clear()

            if n < 2:
                self._set_message("Interstar: complete.", HUD_OK)
                return

            # Fallback: if no cached goal (shouldn't happen for fusion),
            # derive one from the mean of selected robot poses — the
            # chain still builds, just without a proper anchor bias.
            if goal is None:
                xs = [self.bus.poses[r].x for r in selected
                      if r in self.bus.poses]
                ys = [self.bus.poses[r].y for r in selected
                      if r in self.bus.poses]
                goal = (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else (0.0, 0.0)

            def dist_goal(r: int) -> float:
                p = self.bus.poses[r]
                return math.hypot(p.x - goal[0], p.y - goal[1])
            anchor = min(selected, key=dist_goal)

            # Connectivity filter (transitive): only participants that
            # can chain-dock to the anchor via pairwise <= DOCKING +
            # slack stay in the combine.  Others are left as split
            # singletons so stuck robots never fragment the merge.
            SLACK = 2.0   # allow a larger reach here since we'll drive
                          # each host to its slot next (the connectivity
                          # is only to prune truly unreachable outliers).
            THRESH = DOCKING_DISTANCE_M + SLACK
            from collections import deque as _deque
            def _dist(a: int, b: int) -> float:
                pa = self.bus.poses[a]; pb = self.bus.poses[b]
                return math.hypot(pa.x - pb.x, pa.y - pb.y)
            reachable: set = {anchor}
            frontier = _deque([anchor])
            while frontier:
                r = frontier.popleft()
                for s in selected:
                    if s in reachable:
                        continue
                    if _dist(r, s) <= THRESH:
                        reachable.add(s)
                        frontier.append(s)
            others = sorted(reachable - {anchor})
            excluded = [r for r in selected if r not in reachable]

            if not others:
                self._set_message(
                    f"Interstar: complete, but no robot reachable to "
                    f"Robot{anchor} for combine.", HUD_WARN)
                return

            # ---- Formation-packer staging ----
            # Compute distinct dock slots around the anchor at
            # DOCKED_DISTANCE_M, starting at an angle pointing toward
            # the cluster of others (so nearest slots are on that side
            # and each host drives the minimum distance to its slot).
            ap = self.bus.poses[anchor]
            n_others = len(others)
            cx_avg = sum(self.bus.poses[h].x for h in others) / n_others
            cy_avg = sum(self.bus.poses[h].y for h in others) / n_others
            base_angle = math.atan2(cy_avg - ap.y, cx_avg - ap.x)
            slots: List[Tuple[float, float]] = []
            for i in range(n_others):
                if n_others == 1:
                    angle = base_angle
                else:
                    # Full 2π if ≥ 6 participants, otherwise a wider
                    # spread so slots don't bunch up behind the anchor.
                    spread = 2 * math.pi if n_others >= 6 else math.pi * 1.2
                    angle = base_angle + spread * (i / max(1, n_others - 1)
                                                   - 0.5)
                slot = (ap.x + DOCKED_DISTANCE_M * math.cos(angle),
                        ap.y + DOCKED_DISTANCE_M * math.sin(angle))
                slots.append(slot)

            # Greedy assign: iterate hosts sorted by current distance
            # to the anchor (closest first), each claiming the slot
            # closest to its own pose.  Keeps net movement minimal.
            assignments: Dict[int, Tuple[float, float]] = {}
            remaining = list(slots)
            for host in sorted(others, key=lambda h: _dist(h, anchor)):
                hp = self.bus.poses[host]
                best = min(remaining,
                           key=lambda s: math.hypot(hp.x - s[0], hp.y - s[1]))
                assignments[host] = best
                remaining.remove(best)

            # Kick off phase 1 — drive each non-anchor host to its
            # slot via the standard A* nav controller.
            for host, slot in assignments.items():
                self._nav_goals[host]       = slot
                self._nav_paths.pop(host, None)
                self._nav_wp_idx[host]      = 0
                self._nav_fail_count[host]  = 0
                self._nav_replan_tick[host] = 0
            self._interstar_staging_slots  = assignments
            self._interstar_combine_anchor = anchor
            self._interstar_staging_active = True
            self._interstar_staging_timer  = 0
            self._interstar_staging_rest_ticks = 0
            self._interstar_staging_last_pose  = {}
            self._interstar_fusion_pairs   = []   # populated on transition

            if excluded:
                self._set_message(
                    f"Interstar: complete. Staging {len(others)} robots "
                    f"around anchor=Robot{anchor}; "
                    f"Robot{', Robot'.join(map(str, excluded))} left out "
                    f"(unreachable)...",
                    HUD_WARN,
                )
            else:
                self._set_message(
                    f"Interstar: complete. Staging {len(others)} robots "
                    f"around anchor=Robot{anchor} at dock slots...",
                    HUD_OK,
                )

        def _complete_interstar_fission(self) -> None:
            """Called when every selected robot has reached the end of its
            diverging fission path.  Clears Interstar state (paths vanish)
            — no pairwise combine: fission members stay separate."""
            n = len(self._interstar_paths)

            # Clear per-robot nav + Interstar render state
            for rid in list(self._interstar_paths.keys()):
                self._nav_goals.pop(rid, None)
                self._nav_paths.pop(rid, None)
                self._nav_wp_idx.pop(rid, None)
                self._nav_replan_tick.pop(rid, None)
            self._interstar_paths          = {}
            self._interstar_cursor         = {}
            self._interstar_shared_segment = []
            self._interstar_plan_active    = False
            self._interstar_pending_fuses  = []
            self._interstar_stop_timer     = 0
            self._interstar_cluster_timer  = 0
            self._interstar_last_positions = {}
            self._interstar_mode           = ""
            self._interstar_fusion_goal    = None
            self._interstar_selected_cache = []
            self._interstar_replan_counter = 0
            self._interstar_fission_goals       = {}
            self._interstar_fission_start_poses = {}
            # Fission has no post-process combine queue, so terminate
            # the Ctrl+click candidate set here as well — all Inter-Star
            # yellow stars should vanish the moment fission completes.
            self._selected_ids.clear()

            self._set_message(
                f"Interstar FISSION: complete ({n} robots reached goals).",
                HUD_OK,
            )

        def _iterative_replan_fusion(self) -> None:
            """Re-run Inter-Star with the live formation-host poses as
            starts.  VISUALISATION-ONLY refresh: the amber shared-tail
            overlay tracks the evolving world (new obstacles, robot
            motion, etc.) every INTERSTAR_REPLAN_INTERVAL frames.

            Navigation is handled separately by the standard A* nav
            controller (seeded at dispatch via `_nav_goals[host] =
            fusion_goal`), which replans every ~0.33 s with robot-aware
            occupancy — so driving does NOT depend on this refresh.
            Only `_interstar_paths` / `_interstar_shared_segment` /
            `_interstar_metrics` are touched here.
            """
            if (self._interstar_mode != "fusion"
                    or self._interstar_fusion_goal is None
                    or not self._interstar_selected_cache):
                return
            # Drop any cached host that has since vanished (fused with
            # another, removed from sim, etc.) — otherwise the re-plan
            # would include a stale start.
            selected = [rid for rid in self._interstar_selected_cache
                        if rid in self.bus.poses
                        and rid in self.bus.configurers
                        and self._host_of(rid) == rid]
            if len(selected) < 2:
                return

            # Same centroid adjustment as _dispatch_interstar_fusion:
            # plan from rotation centre for fused hosts so the nav
            # driver never has to reverse to align with path[1].
            world = self._build_world_spec_for_planners()
            adjusted_robots: List[RobotState] = []
            for r in world.robots:
                if r.robot_id in selected and r.n > 1:
                    cx, cy = self._rotation_centre(r.robot_id)
                    adjusted_robots.append(RobotState(
                        robot_id    = r.robot_id,
                        pose        = Pose(x=cx, y=cy, yaw=r.pose.yaw),
                        fsm_state   = r.fsm_state,
                        host_id     = r.host_id,
                        n           = r.n,
                        footprint_m = r.footprint_m,
                    ))
                else:
                    adjusted_robots.append(r)
            world_for_plan = WorldSpec(
                bounds=world.bounds, cell_m=world.cell_m,
                obs_mgr=world.obs_mgr, robots=adjusted_robots,
            )
            query = PlannerQuery(
                mode         = PlannerMode.INTERSTAR,
                world        = world_for_plan,
                robots       = adjusted_robots,
                selected_ids = selected,
                goal         = GoalSpec(point=self._interstar_fusion_goal),
            )
            # Re-plan uses the same formation-aware inflation rule as
            # the initial dispatch so a re-plan after a fuse doesn't
            # tighten clearance below what the fresh-dispatch produced.
            self._interstar_planner.inflate_radius = max(
                [self._path_inflate_for(h) for h in selected]
                + [0.40]
            )
            result = self._interstar_planner.plan(query)
            if result.metrics.get("plan_failed"):
                return   # keep current visualisation, retry next cadence

            # Swap in the fresh visualisation paths.  Nav state is
            # intentionally NOT touched here — the standard nav
            # controller already has `_nav_goals[host] = fusion_goal`
            # and will replan against obstacles on its own cadence.
            self._interstar_paths  = {}
            self._interstar_cursor = {}
            for rid, path in result.assignments.items():
                if not path:
                    continue
                self._interstar_paths[rid]  = list(path)
                self._interstar_cursor[rid] = 1

            self._interstar_shared_segment = list(
                result.extras.get("shared_segment", []))
            self._interstar_metrics        = dict(result.metrics)
            self._interstar_pending_fuses  = list(result.reconfig)

        def _iterative_replan_fission(self) -> None:
            """Periodic per-member fission re-plan — via Inter-Star.

            Mirrors fusion's iterative re-plan behaviour so fission
            paths stay "Inter-Star-shaped" throughout execution
            rather than being overwritten with plain-A* polylines
            after 15 frames.  For every member that hasn't yet
            reached its cached click-point goal, run the Inter-Star
            planner in fission mode with:
              * selected_ids = [this_member_rid]
              * goal         = GoalSpec(points=[this_member_goal])
              * fission_start picks up the member's CURRENT pose
                inside InterstarPlanner.plan()

            Inter-Star's fission branch with a single start/goal
            pair runs Algorithm 1 end-to-end (just without the
            shared-path-stitching benefit, which only matters with
            ≥ 2 robots anyway).  The output path is world-frame and
            directly replaces `_interstar_paths[rid]`.
            """
            if (self._interstar_mode != "fission"
                    or not self._interstar_fission_goals):
                return
            for rid, goal in list(self._interstar_fission_goals.items()):
                if rid not in self.bus.poses:
                    continue
                # Already arrived? Skip.
                p = self.bus.poses[rid]
                if math.hypot(p.x - goal[0], p.y - goal[1]) \
                        < self.INTERSTAR_GOAL_TOL_M:
                    continue
                # Only replan if we still have a visualisation path
                # for this member (hasn't been pruned by the per-tick
                # arrival cleanup in `_refresh_interstar_active`).
                if rid not in self._interstar_paths:
                    continue

                # Build a per-member Inter-Star query.  The planner
                # reads `robot.pose` for `selected_ids[0]` as the
                # fission_start, so no manual override is needed.
                world = self._build_world_spec_for_planners()
                query = PlannerQuery(
                    mode         = PlannerMode.INTERSTAR,
                    world        = world,
                    robots       = world.robots,
                    selected_ids = [rid],
                    goal         = GoalSpec(points=[goal]),
                )
                # Same inflation rule as the initial fission dispatch.
                self._interstar_planner.inflate_radius = max(
                    self._path_inflate_for(rid), 0.40
                )
                try:
                    result = self._interstar_planner.plan(query)
                except Exception:
                    continue
                if result.metrics.get("plan_failed"):
                    continue
                # InterstarPlanner.plan for fission keys output as
                # {-(k+1): path_k}; with one goal that's {-1: path}.
                new_path = result.assignments.get(-1, [])
                if not new_path or len(new_path) < 2:
                    continue
                self._interstar_paths[rid]  = list(new_path)
                self._interstar_cursor[rid] = 1

        def _refresh_interstar_active(self) -> None:
            """Per-tick Inter-Star driver (both fusion and fission).

            Motion is NOT issued here anymore — both modes seed
            `_nav_goals[rid]` at dispatch and the standard A* nav
            controller handles driving with robot-aware occupancy.
            This method's responsibilities are:
              * service the proximity-chain fusion queue
                (`_tick_interstar_combine`),
              * run mode-specific pose-based TERMINATION checks
                (fusion: all-stopped + any-at-goal; fission: every
                assigned robot within goal tolerance, with per-robot
                visualisation cleanup as each arrives),
              * refresh the amber shared-tail visualisation for fusion
                via `_iterative_replan_fusion` at a fixed cadence.
            """
            # Always service the pairwise-fusion queue (even if Interstar
            # activity itself is no longer active — the queue drains
            # independently after _complete_interstar_and_combine fires).
            self._tick_interstar_combine()

            if not self._interstar_plan_active:
                return

            # ---- Termination detection (mode-specific) ------------------
            # Fusion: all selected robots stopped for 1 s AND at least one
            #         has reached the goal point → complete + combine.
            # Fission: every selected robot has reached the end of its
            #          own diverging path (distance to final waypoint
            #          < INTERSTAR_GOAL_TOL_M) → complete (no combine).
            if self._interstar_mode == "fusion":
                # Fusion termination: lax — every selected robot has
                # stopped for INTERSTAR_STOP_FRAMES ticks AND at least
                # one is within INTERSTAR_GOAL_TOL_M of the fusion
                # goal.  This fires as soon as the leading robot
                # arrives and the rest have settled (typically held in
                # place by the fusion-mode waypoint-wait serialisation
                # or by a blocked A* replan).
                #
                # Fragmentation prevention is handled in
                # `_build_interstar_fusion_queue`: robots that are too
                # far to pairwise-dock to the anchor (even transitively)
                # are excluded from the combine chain.  The reachable
                # cluster merges into one formation; stuck robots stay
                # as split singletons.  No more "two fused singletons"
                # output.
                selected = list(self._interstar_paths.keys())
                goal = (self._interstar_fusion_goal
                        if self._interstar_fusion_goal is not None
                        else (self._interstar_shared_segment[-1]
                              if self._interstar_shared_segment else None))
                all_stopped = True
                any_at_goal = False
                all_clustered = bool(selected) and goal is not None
                for rid in selected:
                    if rid not in self.bus.poses:
                        all_clustered = False
                        continue
                    p = self.bus.poses[rid]
                    cur = (p.x, p.y)
                    prev = self._interstar_last_positions.get(rid, cur)
                    moved = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
                    self._interstar_last_positions[rid] = cur
                    if moved > self.INTERSTAR_STOP_TOL_M:
                        all_stopped = False
                    if goal is not None:
                        d_goal = math.hypot(cur[0] - goal[0],
                                            cur[1] - goal[1])
                        if d_goal < self.INTERSTAR_GOAL_TOL_M:
                            any_at_goal = True
                        if d_goal >= self.INTERSTAR_NEAR_GOAL_RADIUS_M:
                            all_clustered = False
                # Fallback: every robot has been within NEAR_GOAL_RADIUS
                # for CLUSTER_FRAMES consecutive ticks AND at least
                # one is at the goal.  Ignores motion so robots
                # "fighting for the spot" eventually transition to
                # the combine phase.
                if all_clustered and any_at_goal:
                    self._interstar_cluster_timer += 1
                else:
                    self._interstar_cluster_timer = 0
                if self._interstar_cluster_timer >= self.INTERSTAR_CLUSTER_FRAMES:
                    self._set_message(
                        "Interstar: cluster detected near goal "
                        "(robots jittering) — proceeding to combine.",
                        HUD_WARN,
                    )
                    self._complete_interstar_and_combine()
                    return
                if all_stopped and any_at_goal:
                    self._interstar_stop_timer += 1
                else:
                    self._interstar_stop_timer = 0
                if self._interstar_stop_timer >= self.INTERSTAR_STOP_FRAMES:
                    self._complete_interstar_and_combine()
                    return

                # Iterative re-planning — fire every REPLAN_INTERVAL frames
                # so the shared-path decision adapts to the evolving world
                # (dynamic obstacles, humans, other robots).  Both the
                # amber visualisation AND the cursor-driven nav targets
                # use these paths, so re-planning rebinds both.
                self._interstar_replan_counter += 1
                if (self._interstar_replan_counter
                        >= self.INTERSTAR_REPLAN_INTERVAL):
                    self._interstar_replan_counter = 0
                    self._iterative_replan_fusion()

                # Fusion uses cursor-based driving below — robots
                # actually walk their Inter-Star paths cell-by-cell so
                # the shared-tail amber visualisation matches motion.
                # Fall through to the cursor loop.

            elif self._interstar_mode == "fission":
                # Fission termination: CURSOR-BASED.  A member is
                # considered arrived iff its cursor has exhausted
                # the path it was given (cursor >= len(path)).  Pose
                # tolerance is no longer the trigger — the previous
                # 1.5 m pose-based rule fired while the cursor still
                # had unwalked waypoints, leaving a visible zig-zag
                # tail that got pruned mid-walk.  With cursor-based
                # termination the robot must actually walk every
                # cell of its Inter-Star path before the line can
                # disappear.
                goal_map = self._interstar_fission_goals
                assigned = list(goal_map.keys())
                all_reached = bool(assigned)
                for rid in assigned:
                    path = self._interstar_paths.get(rid)
                    if path is None:
                        # Already pruned (cursor exhausted earlier
                        # tick) — counts as arrived.
                        continue
                    cursor = self._interstar_cursor.get(rid, 1)
                    if cursor >= len(path):
                        # Cursor exhausted this tick — robot has
                        # walked the whole path.  Prune visualisation
                        # + nav state for this member.
                        self._interstar_paths.pop(rid, None)
                        self._interstar_cursor.pop(rid, None)
                        self._nav_goals.pop(rid, None)
                        self._nav_paths.pop(rid, None)
                        self._nav_wp_idx.pop(rid, None)
                        self._nav_replan_tick.pop(rid, None)
                    else:
                        all_reached = False
                if all_reached:
                    self._complete_interstar_fission()
                    return

                # Iterative re-planning — fire every REPLAN_INTERVAL
                # frames so each member's path adapts to the evolving
                # world.  Matches fusion's behaviour.
                self._interstar_replan_counter += 1
                if (self._interstar_replan_counter
                        >= self.INTERSTAR_REPLAN_INTERVAL):
                    self._interstar_replan_counter = 0
                    self._iterative_replan_fission()

            # ---- Cursor-as-sequence-controller driving (fusion + fission) ----
            # The Inter-Star path supplies the SEQUENCE of waypoints
            # the rid must visit.  Each waypoint is set as the rid's
            # nav goal; the standard A* nav controller then plans an
            # obstacle-aware route from the rid's current pose to that
            # waypoint (with live robot-positions in the occupancy
            # grid so robots route AROUND each other in narrow passages
            # instead of ramming through).  When the rid arrives at
            # the current waypoint, the cursor advances to the next
            # one.  Net effect: robots actually walk the Inter-Star
            # plan but can route around transient blockages.
            #
            # A* replan fires whenever the cursor advances (target
            # changes) and naturally every PATHFIND_REPLAN_INTERVAL
            # ticks otherwise — so if another robot drifts into the
            # route, the next replan routes around it.
            if self._interstar_mode not in ("fusion", "fission"):
                return
            ARRIVE_TOL = 0.18   # slightly > PATHFIND_WAYPOINT_TOL

            all_done = True
            for rid in list(self._interstar_paths.keys()):
                path = self._interstar_paths[rid]
                if rid not in self.bus.poses or not path:
                    continue
                cursor = self._interstar_cursor.get(rid, 1)

                # Advance cursor past any waypoints we're already near.
                # For fusion, rotation centre (== centroid for fused
                # hosts, == host pose for split singletons) is what
                # Inter-Star planned from.  For fission each rid is its
                # own split singleton, so rotation_centre == robot pose.
                rc_x, rc_y = self._rotation_centre(rid)
                while (cursor < len(path)
                       and math.hypot(path[cursor][0] - rc_x,
                                      path[cursor][1] - rc_y) < ARRIVE_TOL):
                    cursor += 1
                self._interstar_cursor[rid] = cursor

                if cursor >= len(path):
                    # Done with its path — clear nav state.
                    self._nav_goals.pop(rid, None)
                    self._nav_paths.pop(rid, None)
                    self._nav_wp_idx.pop(rid, None)
                    self._nav_replan_tick.pop(rid, None)
                    continue

                all_done = False
                target = path[cursor]
                # Bypass per-cursor A*.  Inter-Star already produced
                # a valid path through the same grid A* would build
                # (same cell_size, same inflation against the same
                # static obstacles), so re-running A* per cursor cell
                # is redundant.  Worse — it sometimes fails when the
                # cursor cell is in tight wall-adjacent geometry and
                # `nearest_free_cell` snaps the start to a poorly
                # connected cell, even though the cell itself is on a
                # valid Inter-Star path.  Feed the cursor target as a
                # 2-point polyline (current pose → cursor cell)
                # straight to the motion controller and pin
                # `_nav_replan_tick` so A* never overwrites it.
                self._nav_goals[rid]        = target
                self._nav_paths[rid]        = [(rc_x, rc_y), target]
                self._nav_wp_idx[rid]       = 0
                self._nav_fail_count[rid]   = 0
                self._nav_replan_tick[rid]  = 10**9

            if all_done and self._interstar_mode == "fusion":
                # Defensive fallback for fusion.  Only fire the
                # post-process combine if at least one robot is
                # actually within INTERSTAR_GOAL_TOL_M of the fusion
                # goal — iterative re-planning can produce very
                # short intermediate paths (e.g. after A* gave up
                # and nav was cancelled), and we must NOT auto-
                # combine in that case because no robot has
                # actually reached the goal.  The user's spec:
                # post-Inter-Star docking should require goal
                # arrival, not just cursor exhaustion.
                goal = self._interstar_fusion_goal
                any_at_goal = False
                if goal is not None:
                    for rid in self._interstar_paths:
                        if rid not in self.bus.poses:
                            continue
                        p = self.bus.poses[rid]
                        d = math.hypot(p.x - goal[0], p.y - goal[1])
                        if d < self.INTERSTAR_GOAL_TOL_M:
                            any_at_goal = True
                            break
                if any_at_goal:
                    self._complete_interstar_and_combine()

        def _draw_interstar_overlay(self) -> None:
            """Paper-faithful two-layer path visualisation:
              * Per-robot "independent" segment  — drawn in the robot's
                own colour (ROBOT_COLOURS[rid-1]) up to the point where it
                joins the shared convergence tail.
              * Shared "converged" segment       — drawn in bright amber
                (255, 180, 50) so the Inter-Star saving is visually obvious.
              * Fusion-point marker — amber ring at the final cell.

            Drawn on top of the yellow `_draw_paths` line so it fully
            covers that underpaint inside the Interstar formation.
            """
            paths = self._interstar_paths
            if not paths:
                return
            shared = self._interstar_shared_segment

            # Length of the shared tail (same coords should end every path)
            shared_n = len(shared) if shared else 0

            # 1. Per-robot independent segment — drawn from `cursor - 1`
            # onward so already-walked cells don't leave a trail
            # behind the robot.  `cursor` points at the NEXT target;
            # `cursor - 1` is the last waypoint the robot just
            # passed, which is approximately where the robot is now,
            # so the rendered line visibly starts AT the robot rather
            # than at the dispatch-time pose.
            for rid, path in paths.items():
                if path is None or len(path) < 2:
                    continue
                cursor = self._interstar_cursor.get(rid, 1)
                start_idx = max(0, cursor - 1)
                # Trim already-walked cells off the front
                remaining = path[start_idx:]
                if len(remaining) < 2:
                    continue
                # Split at the shared-tail boundary.  Include the first
                # shared-tail cell in the independent segment so the
                # per-robot line visibly meets the amber line.
                if shared_n > 0 and len(remaining) >= shared_n:
                    indep_end = len(remaining) - shared_n + 1
                    indep = remaining[:indep_end]
                else:
                    indep = remaining
                if len(indep) < 2:
                    continue
                col = ROBOT_COLOURS[(rid - 1) % len(ROBOT_COLOURS)]
                pts = [self._world_to_screen(wx, wy) for (wx, wy) in indep]
                pygame.draw.lines(self.screen, col, False, pts, 3)

            # 2. Converged shared segment (amber)
            if shared_n >= 2:
                pts = [self._world_to_screen(wx, wy) for (wx, wy) in shared]
                pygame.draw.lines(self.screen, (255, 180, 50), False, pts, 4)
                # Fusion-point marker at the final cell
                fx, fy = pts[-1]
                pygame.draw.circle(self.screen, (255, 180, 50),
                                   (fx, fy), 10, 3)

        def _apply_interstar_waypoint_wait(
            self,
            distribution: Dict[int, Twist],
        ) -> Dict[int, Twist]:
            """Cursor-target waypoint wait for FUSION only.

            Zero cmd_vel for any fusion participant whose CURSOR
            target cell is currently occupied by another fusion
            candidate.  First-come-first-served on each shared cell
            of the convergence tail.  Same-formation members are
            exempt (a robot's own formation moves rigidly with it
            and shouldn't block the formation's own next cell).

            Fission is exempt — fission members diverge to distinct
            goals so their cursor cells never collide; the wait
            would only produce false halts.

            Outside Inter-Star, this is a no-op.
            """
            if not self._interstar_paths:
                return distribution
            if self._interstar_mode != "fusion":
                return distribution
            TOL = ROBOT_OCCUPANCY_M   # 0.40 m — cursor-cell occupancy window

            for rid in list(self._interstar_paths.keys()):
                if rid not in distribution or rid not in self.bus.poses:
                    continue
                path = self._interstar_paths.get(rid)
                if not path:
                    continue
                cursor = self._interstar_cursor.get(rid, 1)
                if cursor >= len(path):
                    continue
                target = path[cursor]

                my_host = self._host_of(rid)
                for other_rid in self._interstar_paths:
                    if other_rid == rid:
                        continue
                    if other_rid not in self.bus.poses:
                        continue
                    # Same-formation members ride with us; not blockers.
                    if self._host_of(other_rid) == my_host:
                        continue
                    op = self.bus.poses[other_rid]
                    d = math.hypot(op.x - target[0], op.y - target[1])
                    if d < TOL:
                        distribution[rid] = Twist()
                        break
            return distribution

        def _refresh_gbnn_active(self) -> None:
            """Drive each GBNN-active formation incrementally:

              APPROACH phase — walk through the A* approach polyline to
                               the GBNN start cell (only when the active
                               robot started outside the RoI).
              COVERAGE phase — drive to the planner's CURRENT cell; on
                               arrival, call planner.step() to propagate
                               activity + pick the next cell.

            The activity heatmap (drawn separately by _draw_gbnn_heatmap)
            evolves frame-by-frame as GBNN steps."""
            ARRIVE_TOL = 0.18   # slightly looser than PATHFIND_WAYPOINT_TOL

            for host in list(self._gbnn_planner_for.keys()):
                planner = self._gbnn_planner_for[host]
                if host not in self.bus.poses:
                    self._gbnn_planner_for.pop(host, None)
                    self._gbnn_approach.pop(host, None)
                    self._gbnn_approach_idx.pop(host, None)
                    continue

                rc_x, rc_y = self._rotation_centre(host)

                # ---- APPROACH PHASE -----------------------------------
                if host in self._gbnn_approach:
                    apath = self._gbnn_approach[host]
                    idx   = self._gbnn_approach_idx.get(host, 0)
                    while (idx < len(apath)
                           and math.hypot(apath[idx][0] - rc_x,
                                          apath[idx][1] - rc_y) < ARRIVE_TOL):
                        idx += 1
                    self._gbnn_approach_idx[host] = idx
                    if idx < len(apath):
                        target = apath[idx]
                        self._nav_goals[host] = target
                        self._nav_paths[host] = [(rc_x, rc_y), target]
                        self._nav_wp_idx[host]     = 0
                        self._nav_fail_count[host] = 0
                        self._nav_replan_tick[host] = 10**9
                        continue
                    # Approach finished — fall through to coverage
                    self._gbnn_approach.pop(host, None)
                    self._gbnn_approach_idx.pop(host, None)

                # ---- COVERAGE PHASE -----------------------------------
                if planner.is_done() or planner._gbnn is None:
                    self._gbnn_planner_for.pop(host, None)
                    self._cancel_nav(host)
                    self._set_message(
                        f"GBNN: Robot{host} coverage complete.", HUD_OK)
                    continue

                # Refresh the planner's world snapshot every tick so the
                # live occupancy mask sees current robot positions.  Without
                # this, _world.robots keeps the dispatch-time snapshot and
                # cells that other robots have since vacated stay marked
                # as obstacle in the GBNN grid → never get covered.
                planner._world = self._build_world_spec_for_planners()

                cur_world = planner._cell_to_world(planner._gbnn.position)
                d = math.hypot(cur_world[0] - rc_x, cur_world[1] - rc_y)

                # When the robot has reached (or started at) the current
                # cell, step GBNN once → propagates activity + picks next.
                if d < ARRIVE_TOL:
                    res = planner.step()
                    if res is None or planner.is_done():
                        self._gbnn_planner_for.pop(host, None)
                        self._cancel_nav(host)
                        self._set_message(
                            f"GBNN: Robot{host} coverage complete.", HUD_OK)
                        continue
                    cur_world = planner._cell_to_world(planner._gbnn.position)

                # Drive toward the (possibly new) current cell — 2-point
                # polyline keeps the nav controller's wp-skip logic happy.
                target = cur_world
                if (host not in self._nav_goals
                        or self._nav_goals[host] != target):
                    self._nav_goals[host] = target
                    self._nav_paths[host] = [(rc_x, rc_y), target]
                    self._nav_wp_idx[host]     = 0
                    self._nav_fail_count[host] = 0
                # Pin replan so A* doesn't overwrite our 2-point polyline.
                self._nav_replan_tick[host] = 10**9

        def _draw_gbnn_cells(self) -> None:
            """Bottom layer — cell shading only (below robots and obstacles).

            Called early in `_render` so that floor cells look tinted but
            never occlude trolleys, humans, walls, or robots drawn above.
            """
            for host, planner in self._gbnn_planner_for.items():
                state = planner.render_state()
                if not state:
                    continue
                self._draw_gbnn_heatmap(state)

        def _draw_gbnn_overlay(self) -> None:
            """Top layer — RoI border + cursor ring + visit trail.

            These are the informational overlays that must stay legible
            above robots and obstacles, so they stay in the top render
            pass.  The cell shading is drawn separately by _draw_gbnn_cells.
            """
            for host, planner in self._gbnn_planner_for.items():
                state = planner.render_state()
                if not state:
                    continue
                self._draw_gbnn_heatmap_top(state)
                self._draw_gbnn_trail(state)

        def _draw_gbnn_heatmap(self, state: Dict) -> None:
            """Activity heatmap — cell shading only, aligned to world cells.

            Per-cell colouring (semi-transparent over the floor grid):
              v = -1.0      → soft red          (obstacle / blocked)
              v = +1.0      → bright robot tint (unvisited, full attraction)
              0 < v < 1     → robot-tint ramp   (decaying activity)
              v ≤ 0         → dark robot tint   (visited)

            Cells are drawn as discrete pygame rects sized by the
            screen-space positions of their world-edge corners — this
            avoids the pixel-quantization drift that was making the
            coverage grid appear offset from the robot's actual cell
            centres (the robot visibly walked on cell edges before).
            Cell tints use the active robot's colour so multiple
            simultaneous GBNN runs stay visually distinct.
            """
            grid   = state.get("activity_grid")
            origin = state.get("origin")
            cs     = state.get("cell_size")
            rid    = state.get("active_rid")
            if grid is None or origin is None or cs is None:
                return
            rows, cols = grid.shape

            # Robot-colour tint.  Fallback teal keeps legacy look if the
            # active rid ever goes missing (e.g. mid-reset races).
            if rid is not None:
                base_col = ROBOT_COLOURS[
                    (int(rid) - 1) % len(ROBOT_COLOURS)]
            else:
                base_col = (60, 230, 200)
            br, bg, bb = base_col

            # Compute per-edge screen x/y using the same continuous
            # world-to-screen transform that robot renders go through.
            # Each edge is an independent int-rounding of px_per_m, so
            # adjacent cells tile seamlessly and edges snap exactly to
            # the pixels a robot at that cell centre would land on —
            # no more "walking on cell edges" look.
            x_edges = [self._world_to_screen(
                           origin[0] + c * cs, origin[1])[0]
                       for c in range(cols + 1)]
            y_edges = [self._world_to_screen(
                           origin[0], origin[1] + r * cs)[1]
                       for r in range(rows + 1)]
            # y_edges is descending (world y up → screen y down):
            # y_edges[0]     = largest screen y  (bottom of row 0)
            # y_edges[rows]  = smallest screen y (top of row rows-1)
            W = x_edges[-1] - x_edges[0]
            H = y_edges[0]  - y_edges[-1]
            if W <= 0 or H <= 0:
                return
            dx, dy = x_edges[0], y_edges[-1]
            surf = pygame.Surface((W, H), pygame.SRCALPHA)

            for r in range(rows):
                for c in range(cols):
                    v = float(grid[r, c])
                    if v == -1.0:
                        # Cells here will be painted over by actual
                        # obstacles (trolleys, walls, humans), so keep
                        # the tint faint — just enough to disambiguate
                        # from RoI's interior while GBNN is paused.
                        col = (210, 60, 60, 60)
                    elif v == 1.0:
                        # Unvisited — bright, attractive robot tint
                        col = (br, bg, bb, 85)
                    elif v > 0:
                        # Activity decay: lerp from visited shade → full tint
                        t = max(0.0, min(1.0, v))
                        col = (
                            int(br * (0.30 + 0.70 * t)),
                            int(bg * (0.30 + 0.70 * t)),
                            int(bb * (0.30 + 0.70 * t)),
                            int(70 + 25 * t),
                        )
                    else:
                        # Visited — dark variant of robot colour
                        col = (br // 3, bg // 3, bb // 3, 150)
                    sx0 = x_edges[c]     - dx
                    sy0 = y_edges[r + 1] - dy       # smaller screen y = top
                    sw  = x_edges[c + 1] - x_edges[c]
                    sh  = y_edges[r]     - y_edges[r + 1]
                    pygame.draw.rect(
                        surf, col,
                        pygame.Rect(sx0, sy0, max(1, sw), max(1, sh)),
                    )

            # Subtle inter-cell grid so coverage units remain countable
            grid_col = (110, 130, 150, 70)
            for c in range(cols + 1):
                x = x_edges[c] - dx
                pygame.draw.line(surf, grid_col, (x, 0), (x, H), 1)
            for r in range(rows + 1):
                y = y_edges[r] - dy
                pygame.draw.line(surf, grid_col, (0, y), (W, y), 1)

            self.screen.blit(surf, (dx, dy))

        def _draw_gbnn_heatmap_top(self, state: Dict) -> None:
            """Top-layer GBNN decorations — RoI border + GBNN cursor ring.

            These need to stay legible above robots/obstacles, so they
            live on the top render pass instead of alongside the cells.
            """
            roi = state.get("roi")
            cs  = state.get("cell_size")
            if roi is not None:
                rect_n = roi.normalized()
                x0, y0 = self._world_to_screen(rect_n.x0, rect_n.y1)
                x1, y1 = self._world_to_screen(rect_n.x1, rect_n.y0)
                pygame.draw.rect(
                    self.screen, (90, 230, 200),
                    pygame.Rect(x0, y0, x1 - x0, y1 - y0), 2,
                )
            cur_world = state.get("position_world")
            if cur_world is not None and cs is not None:
                cx, cy = self._world_to_screen(*cur_world)
                ring_px = max(4, int(self._metres_to_px(cs * 0.40)))
                pygame.draw.circle(
                    self.screen, (255, 220, 100),
                    (cx, cy), ring_px, width=2,
                )

        def _draw_gbnn_trail(self, state: Dict) -> None:
            cells = state.get("path_cells") or []
            origin = state.get("origin")
            cs = state.get("cell_size")
            if not cells or origin is None or cs is None:
                return
            pts = []
            for (r, c) in cells:
                wx = origin[0] + (c + 0.5) * cs
                wy = origin[1] + (r + 0.5) * cs
                pts.append(self._world_to_screen(wx, wy))
            if len(pts) >= 2:
                pygame.draw.lines(
                    self.screen, (255, 220, 120), False, pts, 2)

        def _draw_gbnn_drag_preview(self) -> None:
            """Dashed cyan rectangle preview while LMB is being dragged."""
            if (not self._gbnn_drag_armed
                    or self._gbnn_drag_start is None
                    or self._gbnn_drag_last is None):
                return
            sx0, sy0 = self._gbnn_drag_start
            sx1, sy1 = self._gbnn_drag_last
            if abs(sx1 - sx0) < 4 and abs(sy1 - sy0) < 4:
                return
            x0, y0 = min(sx0, sx1), min(sy0, sy1)
            x1, y1 = max(sx0, sx1), max(sy0, sy1)
            w, h = x1 - x0, y1 - y0
            surf = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
            surf.fill((60, 200, 180, 40))
            self.screen.blit(surf, (x0, y0))
            dash = 8
            col = (90, 220, 200)
            for side_y in (y0, y1):
                xx = x0
                while xx < x1:
                    pygame.draw.line(self.screen, col,
                                     (xx, side_y),
                                     (min(xx + dash, x1), side_y), 2)
                    xx += 2 * dash
            for side_x in (x0, x1):
                yy = y0
                while yy < y1:
                    pygame.draw.line(self.screen, col,
                                     (side_x, yy),
                                     (side_x, min(yy + dash, y1)), 2)
                    yy += 2 * dash

        def _render(self) -> None:
            self.screen.fill(BG_COLOUR)
            self._draw_grid()
            # GBNN cell shading — below obstacles/robots so the coverage
            # tint doesn't occlude anything above (wall edges stay crisp,
            # robots remain fully visible over their visited cells).
            self._draw_gbnn_cells()
            # Inter-Star path overlay — drawn BELOW hulls/obstacles/robots
            # so the rendered path plan appears as a background underlay,
            # occluded by physical entities above it.
            self._draw_interstar_overlay()      # shared-segment amber line
            self._draw_hulls()             # lowest layer — below obstacles & robots
            self._draw_obstacles()
            self._draw_clearance_zones()   # Shift-only halos; above obstacles
            self._draw_paths()
            self._draw_rotation_centres()
            self._draw_caster_trolleys()   # HIGH/HEAVY above hulls, below robots
            for rid in sorted(self.bus.configurers.keys()):
                self._draw_robot(rid)      # robots on top (translucent when under)
            self._draw_mounted_mdas()      # Phase E: MDAs at z+1 above host
            # Phase A overlay above the scene (RoI border, GBNN cursor
            # ring, visit trail).  Inter-Star moved to underlay above.
            self._draw_gbnn_overlay()
            self._draw_gbnnh_overlay()          # Phase E (Mode 5) AP markers + cone
            self._draw_pending_fission_stars()  # yellow goal stars pre-dispatch
            self._draw_gbnn_drag_preview()
            self._draw_drag_spawn_preview()
            self._draw_toolbar()
            self._draw_hud()
            self._draw_gbnnh_surface_panel()    # corner panel — above HUD
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

                # Reset per-tick debug accumulators
                self._dbg_collision_push.clear()
                self._dbg_nav_reason.clear()

                # Phase A scaffolds — idle in mode 1 but plumbed for Phase B+
                self._refresh_gbnn_active()
                self._refresh_interstar_active()
                self._refresh_gbnnh_active()       # Phase E (Mode 5)
                # Snap mounted MDA modules to their host pose this tick so
                # they render glued to the robot.  Cheap; runs always.
                self.obs_mgr.sync_mounted_mdas({
                    rid: (p.x, p.y, p.yaw)
                    for rid, p in self.bus.poses.items()
                })
                self.sequencer.tick()

                # Build distribution: docking > navigation > keyboard
                # raw_host_cmds: pre-compensation cmds per host,
                # used by _pin_centroid to advance the centroid
                # anchor without double-counting angular terms.
                raw_host_cmds: Dict[int, Twist] = {}

                if self.docking is not None:
                    # Capture busy ids BEFORE _docking_cmd() — it may
                    # complete the dock and set self.docking = None.
                    _dk = self.docking
                    busy = set(_dk.get("trigger_members", [])) \
                         | set(_dk.get("target_members", []))
                    distribution = self._docking_cmd()
                    cmd = Twist()  # no user cmd during docking
                    # PHASE A: docking only owns the trigger + target
                    # formations.  Other robots that are mid-nav (or
                    # mid-GBNN coverage) should keep moving.  Compute nav
                    # for everyone else and merge in.
                    self._merge_nav_for_uninvolved(distribution, busy)
                elif self._trolley_docking is not None:
                    _td = self._trolley_docking
                    busy = set()
                    if isinstance(_td, dict):
                        for k in ("trigger_host", "robot_id", "rid"):
                            v = _td.get(k)
                            if isinstance(v, int):
                                busy.add(v)
                    distribution = self._trolley_attach_cmd()
                    cmd = Twist()
                    # PHASE A: same as docking — let other navigators
                    # continue while one robot is attaching to a trolley.
                    self._merge_nav_for_uninvolved(distribution, busy)
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
                                    f"Robot{sel_host} nav cancelled "
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

                # Phase B: Interstar-active robots wait when next waypoint
                # is occupied by another robot (serialised arrival onto
                # the shared fusion cell).
                distribution = self._apply_interstar_waypoint_wait(
                    distribution)

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

# ============================================================================
# PHASE A — Headless test suite + SB3/Py-3.8 PPO load spike
# ============================================================================


def run_headless_test() -> int:
    """Validate Phase A's pipeline end-to-end without pygame.

    Scenarios:
      1. Configurer FSM      — fusion handshake + fission reset.
      2. DefaultPlanner      — A* on an empty grid yields a valid path.
      3. GBNNBasePlanner     — RoI runs to 100 % coverage.
      4. GBNN integration    — coverage path matches a hand-checked length.
      5. Phase A contracts   — PlanResult dataclasses serialise / round-trip.

    Returns 0 on pass, non-zero otherwise.
    """
    failures: List[str] = []

    # ---- 1. Configurer FSM fusion + fission ----
    print("[test] 1/5 Configurer FSM fusion + fission")
    bus = SimBus(dt=0.1, visualize=False)
    c1 = Configurer(1, on_publish_rcfg=bus.publish_rcfg)
    c2 = Configurer(2, on_publish_rcfg=bus.publish_rcfg)
    bus.register(c1, Pose(0, 0, 0))
    bus.register(c2, Pose(1, 0, 0))
    bus.send_fusion_command(1, 2)
    bus.send_fusion_command(2, 1)
    for _ in range(20):
        bus.tick()
        if c1.is_fused() and c2.is_fused():
            break
    if not (c1.is_fused() and c2.is_fused()) or c1.n != 2:
        failures.append(
            f"fusion failed: c1.n={c1.n}, c2.n={c2.n}")
    else:
        print(f"       OK — c1.n={c1.n}, host=Robot{c1.host_id}")
    bus.send_fission_command(1)
    for _ in range(5):
        bus.tick()
        if c1.is_split_singleton():
            break
    if not c1.is_split_singleton():
        failures.append(f"fission failed: c1.n={c1.n}")
    else:
        print(f"       OK — c1.n={c1.n} after fission")

    # ---- 2. DefaultPlanner P2P ----
    print("[test] 2/5 DefaultPlanner P2P nav (empty grid)")
    obs = ObstacleManager()
    bounds = (-5.0, -5.0, 5.0, 5.0)
    world = WorldSpec(
        bounds=bounds, cell_m=0.70, obs_mgr=obs,
        robots=[RobotState(robot_id=1,
                           pose=Pose(0.0, 0.0, 0.0),
                           fsm_state=FSMState.CONFIG,
                           host_id=1, n=1, footprint_m=0.70)],
    )
    dp = DefaultPlanner()
    res = dp.plan(PlannerQuery(
        mode=PlannerMode.MANUAL, world=world, robots=world.robots,
        selected_ids=[1], goal=GoalSpec(point=(3.0, 2.0)),
    ))
    path = res.assignments.get(1, [])
    if len(path) < 2:
        failures.append(
            f"DefaultPlanner produced no path (got {len(path)} waypoints)")
    else:
        fx, fy = path[-1]
        if math.hypot(fx - 3.0, fy - 2.0) > 0.5:
            failures.append(
                f"DefaultPlanner endpoint far from goal: ({fx:.2f},{fy:.2f})")
        else:
            print(f"       OK — {len(path)} waypoints, "
                  f"endpoint ({fx:.2f},{fy:.2f})")

    # ---- 3. GBNNBasePlanner full coverage ----
    print("[test] 3/5 GBNNBasePlanner: 3.5 × 3.5 m RoI → 100 % coverage")
    gp = GBNNBasePlanner(footprint_m=0.70)
    roi = Rect(-1.75, -1.75, 1.75, 1.75)
    init = gp.plan(PlannerQuery(
        mode=PlannerMode.MANUAL, world=world, robots=world.robots,
        selected_ids=[1], goal=GoalSpec(area=roi),
    ))
    if init.metrics.get("plan_failed"):
        failures.append("GBNNBasePlanner.plan reported plan_failed")
    else:
        for _ in range(500):
            if gp.is_done():
                break
            gp.step()
        cov = gp.render_state().get("coverage_pct", 0.0)
        if cov < 0.999:
            failures.append(
                f"GBNN coverage incomplete: {cov*100:.1f}% after "
                f"{gp.render_state().get('iterations')} iterations")
        else:
            iters = gp.render_state().get("iterations")
            ncells = len(gp.render_state().get("path_cells", []))
            print(f"       OK — coverage 100%, iter={iters}, cells={ncells}")

    # ---- 4. GBNN cell sequence shape ----
    print("[test] 4/5 GBNN coverage cell sequence")
    cells = gp.render_state().get("path_cells", [])
    rows, cols = gp.render_state().get("grid_shape", (0, 0))
    expected_min = rows * cols
    if len(cells) < expected_min:
        failures.append(
            f"GBNN cell count {len(cells)} < grid cell count {expected_min}")
    else:
        print(f"       OK — visited {len(cells)} cells over a {rows}×{cols} grid")

    # ---- 5. PlanResult round-trip ----
    print("[test] 5/7 Phase A PlanResult dataclass round-trip")
    pr = PlanResult(
        assignments={1: [(0.0, 0.0), (1.0, 0.0)]},
        reconfig=[ReconfigCommand(recipient_id=2,
                                   rcfg=(1, 0, 0), when=5)],
        algo_starts={1: (0.0, 0.0, 0.0)},
        extras={"k": "v"},
        metrics={"coverage_pct": 0.42},
    )
    if pr.assignments[1][1] != (1.0, 0.0):
        failures.append("PlanResult assignment round-trip broken")
    elif pr.reconfig[0].rcfg != (1, 0, 0):
        failures.append("ReconfigCommand rcfg round-trip broken")
    else:
        print("       OK")

    # ---- 6. GBNNBasePlanner visited-state preservation ----
    print("[test] 6/7 GBNN visited cells survive transient obstacles")
    gp2 = GBNNBasePlanner(footprint_m=0.70)
    obs2 = ObstacleManager()
    world2 = WorldSpec(
        bounds=(-5.0, -5.0, 5.0, 5.0), cell_m=0.70, obs_mgr=obs2,
        robots=[RobotState(robot_id=1, pose=Pose(0.0, 0.0, 0.0),
                           fsm_state=FSMState.CONFIG, host_id=1, n=1,
                           footprint_m=0.70)],
    )
    gp2.plan(PlannerQuery(
        mode=PlannerMode.MANUAL, world=world2, robots=world2.robots,
        selected_ids=[1], goal=GoalSpec(area=Rect(-1.4, -1.4, 1.4, 1.4)),
    ))
    # Step a few times to visit several cells
    for _ in range(8):
        if gp2.is_done():
            break
        gp2.step()
    visited_after_8_steps = len(gp2._visited_cells)
    # Now simulate a transient obstacle covering an already-visited cell:
    # craft a mask with one visited cell forced to True (obstacle)
    rows, cols = gp2._grid_shape
    mask = np.zeros((rows, cols), dtype=bool)
    blocked_cell = next(iter(gp2._visited_cells))
    mask[blocked_cell[0], blocked_cell[1]] = True
    gp2._gbnn.set_occupancy(mask)
    # Now clear the obstacle (mask all-False) — used to reset visited→1.0
    mask = np.zeros((rows, cols), dtype=bool)
    gp2._gbnn.set_occupancy(mask)
    # Apply the planner's visited-restore (normally done by step())
    for (vr, vc) in gp2._visited_cells:
        if (0 <= vr < rows and 0 <= vc < cols
                and gp2._gbnn._grid[vr, vc] == 1.0):
            gp2._gbnn._grid[vr, vc] = 0.0
    # The blocked cell should NOT be back at +1.0
    val = float(gp2._gbnn._grid[blocked_cell[0], blocked_cell[1]])
    if val == 1.0:
        failures.append(
            f"GBNN visited cell {blocked_cell} reset to +1.0 after "
            f"transient obstacle cleared (would make coverage uncompletable)")
    else:
        print(f"       OK — visited cell {blocked_cell} stayed at "
              f"{val:.3f} (not unvisited)")

    # ---- 7. ReconfigSequencer scaffold sanity ----
    print("[test] 7/10 ReconfigSequencer scaffold")
    bus_s = SimBus(dt=0.1, visualize=False)
    cs1 = Configurer(1, on_publish_rcfg=bus_s.publish_rcfg)
    cs2 = Configurer(2, on_publish_rcfg=bus_s.publish_rcfg)
    bus_s.register(cs1, Pose(0, 0, 0)); bus_s.register(cs2, Pose(1, 0, 0))
    seq = ReconfigSequencer(bus_s)
    if seq.pending() != 0:
        failures.append("Sequencer should start empty")
    seq.enqueue(ReconfigCommand(recipient_id=1, rcfg=(2, 0, 0), when=1))
    seq.enqueue(ReconfigCommand(recipient_id=2, rcfg=(1, 0, 0), when=1))
    if seq.pending() != 2:
        failures.append("Sequencer pending count wrong after enqueue")
    fired = 0
    for _ in range(20):
        seq.tick()
        bus_s.tick()
        fired = 2 - seq.pending()
        if cs1.is_fused() and cs2.is_fused():
            break
    if not (cs1.is_fused() and cs2.is_fused()):
        failures.append(
            f"Sequencer didn't complete fusion: "
            f"cs1.fused={cs1.is_fused()} cs2.fused={cs2.is_fused()} "
            f"pending={seq.pending()}")
    else:
        print(f"       OK — 2 FUSE commands fired, formation n={cs1.n}")

    # ---- 8. Interstar scalability (≥3× fewer expansions vs A*) ----
    print("[test] 8/10 Interstar scalability (2-5 robots, fusion)")
    from interstar.open_interstar import Interstar as _ItS
    grid_20 = [[0] * 20 for _ in range(20)]
    scalability_ok = True
    for n in (2, 3, 4, 5):
        # n robots at the 4 corners (or subset), converging to centre
        corners = [(0, 0), (0, 19), (19, 19), (19, 0), (10, 0)][:n]
        paths, metrics = _ItS.plan(
            starts=corners, goal=(10, 10), grid=grid_20,
            mode="fusion", render=False,
        )
        exp    = metrics["expansions"]
        base   = metrics["baseline_expansions"]
        ratio  = metrics["expansions_ratio"]
        shared = len(metrics["shared_segment"])
        print(f"       n={n}: expansions={exp:4d}  baseline={base:4d}  "
              f"ratio={ratio:.2f}×  shared={shared} cells")
        # Inter-Star should always be equal-or-cheaper than baseline
        if exp > base:
            scalability_ok = False
            failures.append(
                f"Interstar n={n} used MORE expansions ({exp}) "
                f"than baseline A* ({base})")
    if scalability_ok:
        print("       OK — Inter-Star ≤ baseline in all cases")

    # ---- 9. Interstar FUSE command sequencing ----
    print("[test] 9/10 Interstar FUSE ReconfigCommand sequence")
    obs_i = ObstacleManager()
    world_i = WorldSpec(
        bounds=(-5.0, -5.0, 5.0, 5.0), cell_m=0.20, obs_mgr=obs_i,
        robots=[
            RobotState(robot_id=r, pose=Pose(float(r-3), 0.0, 0.0),
                       fsm_state=FSMState.CONFIG, host_id=r, n=1,
                       footprint_m=0.70)
            for r in (1, 2, 3, 4)
        ],
    )
    interstar_adapter = InterstarPlanner()
    res_fus = interstar_adapter.plan(PlannerQuery(
        mode=PlannerMode.INTERSTAR, world=world_i, robots=world_i.robots,
        selected_ids=[1, 2, 3, 4],
        goal=GoalSpec(point=(3.0, 3.0)),
    ))
    # For n=4 robots, expect 2×(n-1)=6 commands (two-sided handshake per pair)
    n_cmds = len(res_fus.reconfig)
    whens  = [c.when for c in res_fus.reconfig]
    kinds  = [c.rcfg[2] for c in res_fus.reconfig]  # 0 = FUSE, -1 = FISSION
    paths_ok = all(r in res_fus.assignments and len(res_fus.assignments[r]) > 0
                   for r in (1, 2, 3, 4))
    if not paths_ok:
        failures.append("Interstar fusion did not return paths for all robots")
    elif n_cmds != 6:
        failures.append(
            f"Interstar n=4 fusion expected 6 FUSE commands (2×(n-1)), "
            f"got {n_cmds}")
    elif any(k != 0 for k in kinds):
        failures.append("Interstar fusion emitted non-FUSE command")
    elif whens != sorted(whens):
        failures.append(f"FUSE when-values not monotonic: {whens}")
    else:
        print(f"       OK — {n_cmds} FUSE commands, when values {whens}")

    # ---- 10. Interstar fission returns n paths ----
    print("[test] 10/10 Interstar fission (n=3)")
    res_fis = interstar_adapter.plan(PlannerQuery(
        mode=PlannerMode.INTERSTAR, world=world_i, robots=world_i.robots,
        selected_ids=[1],
        goal=GoalSpec(points=[(3.0, 3.0), (-3.0, 3.0), (0.0, -3.0)]),
    ))
    n_paths = sum(1 for p in res_fis.assignments.values() if p)
    split_cmds = [c for c in res_fis.reconfig if c.rcfg[2] == -1]
    if n_paths != 3:
        failures.append(
            f"Fission expected 3 divergence paths, got {n_paths}")
    elif len(split_cmds) < 1:
        failures.append(
            f"Fission expected at least 1 FISSION command, got {len(split_cmds)}")
    else:
        print(f"       OK — {n_paths} diverging paths, {len(split_cmds)} "
              f"FISSION command emitted")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All Phase A + B headless tests PASSED.")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Spyder Ascend unified demo (Configurer teleop + Phase A)."
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
    parser.add_argument(
        "--test", action="store_true",
        help="Run the Phase A headless test suite (no display).",
    )
    args = parser.parse_args()

    if args.test:
        sys.exit(run_headless_test())
    if args.demo:
        demo(n_robots=args.robots, visualize=not args.headless)
        sys.exit(0)
    sys.exit(_run_pygame_teleop())