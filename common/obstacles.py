"""
obstacles.py — Obstacle system for the Configurer Pygame teleop sandbox.

Three obstacle categories:
  StaticMoveable   : table, chair, high_trolley, low_trolley
  StaticImmovable  : wall, pillar
  DynamicObstacle  : sliding_door, pivot_door, human

Each obstacle has a world-frame position, size, and draws itself
using simple geometric shapes with recognisable features.

Dynamic obstacles use a bounded random walk every tick.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Tuple, TYPE_CHECKING
)

if TYPE_CHECKING:
    import pygame                       # only for type hints

# ============================================================================
#  Obstacle type registry
# ============================================================================

class ObstacleKind(Enum):
    # Static — moveable
    TABLE         = auto()
    CHAIR         = auto()
    HIGH_TROLLEY  = auto()
    LOW_TROLLEY   = auto()
    HEAVY_TROLLEY = auto()
    # Static — immovable
    WALL          = auto()      # click-drag wall (variable length)
    PILLAR        = auto()
    # Dynamic
    SLIDING_DOOR  = auto()
    PIVOT_DOOR    = auto()
    HUMAN         = auto()
    # Robot-mountable accessory (Phase E / Pressing 5)
    MDA_MODULE    = auto()      # mobile dual-arm module — pre-mount free
                                # obstacle, post-mount attached to a robot


# ---- sizing (half-widths in metres, height is visual hint only) -----------
# (half_w, half_h) define the rectangular bounding box in world coords.
#
# Trolley sizing rationale (robot full-width=0.70m, caster_dia=0.10m):
#   LOW_TROLLEY:   same footprint as one robot (0.35 × 0.35)
#   HIGH_TROLLEY:  len = caster + 2×robot + caster + clearance
#                       = 0.10 + 1.40 + 0.10 + 0.10 = 1.70 → half 0.85
#                  wid = caster + robot + caster + clearance
#                       = 0.10 + 0.70 + 0.10 + 0.10 = 1.00 → half 0.50
#   HEAVY_TROLLEY: len = caster + 1×robot + caster + clearance
#                       = 0.10 + 0.70 + 0.10 + 0.10 = 1.00 → half 0.50
#                  wid = same as HIGH = 1.00 → half 0.50

OBSTACLE_DIMS: Dict[ObstacleKind, Tuple[float, float]] = {
    ObstacleKind.TABLE:         (0.60, 0.40),
    ObstacleKind.CHAIR:         (0.25, 0.25),
    ObstacleKind.HIGH_TROLLEY:  (0.85, 0.50),
    ObstacleKind.LOW_TROLLEY:   (0.35, 0.35),
    ObstacleKind.HEAVY_TROLLEY: (0.50, 0.50),
    ObstacleKind.WALL:          (1.50, 0.10),   # default; half_w overridden per instance
    ObstacleKind.PILLAR:        (0.20, 0.20),
    ObstacleKind.SLIDING_DOOR:  (0.80, 0.08),
    ObstacleKind.PIVOT_DOOR:    (0.60, 0.08),
    ObstacleKind.HUMAN:         (0.20, 0.20),
    ObstacleKind.MDA_MODULE:    (0.30, 0.30),   # mountable accessory; mount
                                                # inflates host footprint to
                                                # MDA_ARM_RADIUS
}

# ---- MDA-specific config -------------------------------------------------
# Arm reach + FOV cone parameters used by Pressing 5 (GBNN+H surface
# cleaning).  The FoV cone radius was reduced from 3.0 m to 1.2 m
# (40% of the original strategy-doc value) — better matches the
# physical reach of the Mobile Dual-Arm and prevents APs from
# pulling in obstacles that are too far to actually clean.
# This single constant feeds both the FoV cone radius used by
# `_compute_roi_segments_for_ap` and the AP-placement "near an
# obstacle" check, so the two stay consistent.
MDA_ARM_RADIUS: float    = 0.50    # post-mount footprint inflation (metres)
MDA_ARM_REACH_M: float   = 1.2     # FOV cone reach (was 3.0)
MDA_ARM_FOV_DEG: float   = 120.0   # FOV cone angular extent
MDA_MOUNT_RANGE_M: float = 1.50    # max distance for `0` key to mount
                                   # an unmounted module onto a robot

# ---- caster wheel config for HIGH / HEAVY trolleys -----------------------
# Robots can pass under the body; only the 4 corner caster circles block.
CASTER_RADIUS: float = 0.05   # diameter 0.10m

# Trolley-specific properties
#   weight_class: minimum formation size to move the trolley
TROLLEY_WEIGHT_CLASS: Dict[ObstacleKind, int] = {
    ObstacleKind.LOW_TROLLEY:   1,   # singleton can move
    ObstacleKind.HIGH_TROLLEY:  1,   # singleton can move
    ObstacleKind.HEAVY_TROLLEY: 2,   # need 2+ robots fused
}

# Which trolleys use caster-only collision (robots pass under body)
_CASTER_TROLLEYS = {ObstacleKind.HIGH_TROLLEY, ObstacleKind.HEAVY_TROLLEY}

# ---- colours (fill, outline) --------------------------------------------

OBSTACLE_COLOURS: Dict[ObstacleKind, Tuple[Tuple[int,int,int],
                                            Tuple[int,int,int]]] = {
    ObstacleKind.TABLE:         ((139, 90, 43),   (100, 60, 20)),
    ObstacleKind.CHAIR:         ((160, 110, 60),  (110, 70, 30)),
    ObstacleKind.HIGH_TROLLEY:  ((100, 100, 110), (70, 70, 80)),
    ObstacleKind.LOW_TROLLEY:   ((120, 120, 130), (80, 80, 90)),
    ObstacleKind.HEAVY_TROLLEY: ((80, 80, 95),    (55, 55, 70)),
    ObstacleKind.WALL:          ((90, 90, 100),   (60, 60, 70)),
    ObstacleKind.PILLAR:        ((110, 110, 120), (75, 75, 85)),
    ObstacleKind.SLIDING_DOOR:  ((70, 130, 180),  (50, 100, 150)),
    ObstacleKind.PIVOT_DOOR:    ((80, 140, 120),  (55, 110, 90)),
    ObstacleKind.HUMAN:         ((220, 170, 130), (180, 130, 90)),
    ObstacleKind.MDA_MODULE:    ((220, 200, 80),  (160, 140, 30)),
}

# ---- category helpers ----------------------------------------------------

_MOVEABLE = {
    ObstacleKind.TABLE, ObstacleKind.CHAIR,
    ObstacleKind.HIGH_TROLLEY, ObstacleKind.LOW_TROLLEY,
    ObstacleKind.HEAVY_TROLLEY,
    ObstacleKind.MDA_MODULE,    # pre-mount only; ignored when mounted
}
_TROLLEY = {
    ObstacleKind.HIGH_TROLLEY, ObstacleKind.LOW_TROLLEY,
    ObstacleKind.HEAVY_TROLLEY,
}
_IMMOVABLE = {
    ObstacleKind.WALL, ObstacleKind.PILLAR,
    ObstacleKind.SLIDING_DOOR, ObstacleKind.PIVOT_DOOR,
    ObstacleKind.HUMAN,
}
_DYNAMIC = set()   # kept for API compat; all non-moveable are now _IMMOVABLE
# Obstacles that need per-tick updates (animation / random walk)
_TICKED = {
    ObstacleKind.SLIDING_DOOR, ObstacleKind.PIVOT_DOOR,
    ObstacleKind.HUMAN,
}
# Obstacle types spawned via click-drag
_DRAG_SPAWN = {
    ObstacleKind.WALL,          # variable length + orientation
    ObstacleKind.SLIDING_DOOR,  # fixed length, orientation from drag
    ObstacleKind.PIVOT_DOOR,    # fixed length, orientation from drag
    ObstacleKind.PILLAR,        # variable radius from drag
}

def snap_angle_45(angle: float) -> float:
    """Snap an angle (radians) to the nearest multiple of 45°."""
    step = math.pi / 4   # 45°
    return round(angle / step) * step

# Display names (for toolbar and labels)
OBSTACLE_LABELS: Dict[ObstacleKind, str] = {
    ObstacleKind.TABLE:         "Table",
    ObstacleKind.CHAIR:         "Chair",
    ObstacleKind.HIGH_TROLLEY:  "H.Trolley",
    ObstacleKind.LOW_TROLLEY:   "L.Trolley",
    ObstacleKind.HEAVY_TROLLEY: "Hv.Trolley",
    ObstacleKind.WALL:          "Wall",
    ObstacleKind.PILLAR:        "Pillar",
    ObstacleKind.SLIDING_DOOR:  "S.Door",
    ObstacleKind.PIVOT_DOOR:    "P.Door",
    ObstacleKind.HUMAN:         "Human",
    ObstacleKind.MDA_MODULE:    "MDA",
}


# ============================================================================
#  Obstacle dataclass
# ============================================================================

_next_obs_id: int = 0

def _gen_id() -> int:
    global _next_obs_id
    _next_obs_id += 1
    return _next_obs_id


@dataclass
class Obstacle:
    kind:  ObstacleKind
    x:     float = 0.0
    y:     float = 0.0
    yaw:   float = 0.0                     # radians
    uid:   int   = field(default_factory=_gen_id)

    # ---- trolley pinned flag (set by Configurer at reconfiguration) ----
    # When True, the trolley is treated as immovable (attached but the
    # formation is too small to move it).  Flipped on fusion/fission.
    pinned: bool = field(default=False, repr=False)

    # ---- dynamic walk state ----
    _walk_target_x: float = field(default=0.0, repr=False)
    _walk_target_y: float = field(default=0.0, repr=False)
    _walk_timer:    int   = field(default=0,   repr=False)

    # ---- pivot door state ----
    _pivot_angle:    float = field(default=0.0, repr=False)
    _pivot_dir:      int   = field(default=1,   repr=False)
    _pivot_base_yaw: float = field(default=0.0, repr=False)

    # ---- sliding door state ----
    _slide_offset:  float = field(default=0.0, repr=False)
    _slide_dir:     int   = field(default=1,   repr=False)

    # ---- door closed position (for proximity-based doors) ----
    _closed_x:      float = field(default=0.0, repr=False)
    _closed_y:      float = field(default=0.0, repr=False)
    _door_open:     bool  = field(default=False, repr=False)

    # ---- per-instance size override (for variable-length walls / pillars) ----
    _custom_half_w: Optional[float] = field(default=None, repr=False)
    _custom_half_h: Optional[float] = field(default=None, repr=False)

    # ---- MDA mount state (Phase E / Pressing 5) ------------------------
    # When this obstacle is an MDA_MODULE, host_robot_id is None pre-mount
    # and set to the host's robot_id post-mount.  Mounted MDAs are excluded
    # from collision/occupancy/drag — the host robot's footprint inflates
    # to MDA_ARM_RADIUS instead — and they render at z+1 above the host.
    host_robot_id: Optional[int] = field(default=None)

    # ---- cached ----
    @property
    def half_w(self) -> float:
        if self._custom_half_w is not None:
            return self._custom_half_w
        return OBSTACLE_DIMS[self.kind][0]

    @property
    def half_h(self) -> float:
        if self._custom_half_h is not None:
            return self._custom_half_h
        return OBSTACLE_DIMS[self.kind][1]

    @property
    def is_moveable(self) -> bool:
        return self.kind in _MOVEABLE

    @property
    def is_immovable(self) -> bool:
        return self.kind in _IMMOVABLE

    @property
    def is_dynamic(self) -> bool:
        return self.kind in _DYNAMIC

    @property
    def is_trolley(self) -> bool:
        return self.kind in _TROLLEY

    @property
    def is_caster_trolley(self) -> bool:
        """HIGH and HEAVY trolleys: robots pass under body, only casters block."""
        return self.kind in _CASTER_TROLLEYS

    @property
    def is_mda_module(self) -> bool:
        return self.kind == ObstacleKind.MDA_MODULE

    @property
    def is_mounted(self) -> bool:
        """True if this is a mounted MDA module (host_robot_id set)."""
        return self.is_mda_module and self.host_robot_id is not None

    @property
    def weight_class(self) -> int:
        """Minimum formation size required to move this trolley (0 = not a trolley)."""
        return TROLLEY_WEIGHT_CLASS.get(self.kind, 0)

    @property
    def is_draggable(self) -> bool:
        return True     # all obstacles can be dragged per user spec

    @property
    def label(self) -> str:
        return OBSTACLE_LABELS[self.kind]

    def caster_circles(self) -> List[Tuple[float, float, float]]:
        """Return world-frame (cx, cy, radius) for each caster wheel.
        Only meaningful for caster trolleys (HIGH/HEAVY). Returns empty
        list for other obstacle types."""
        if not self.is_caster_trolley:
            return []
        hw, hh = self.half_w, self.half_h
        cr = CASTER_RADIUS
        # Casters inset by caster_radius from each corner
        local_corners = [
            (-hw + cr,  -hh + cr),
            ( hw - cr,  -hh + cr),
            ( hw - cr,   hh - cr),
            (-hw + cr,   hh - cr),
        ]
        co = math.cos(self.yaw)
        so = math.sin(self.yaw)
        result = []
        for lx, ly in local_corners:
            wx = self.x + co * lx - so * ly
            wy = self.y + so * lx + co * ly
            result.append((wx, wy, cr))
        return result

    # ---- axis-aligned bounding box in world frame -----------------------
    def aabb(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) in world coords,
        accounting for rotation."""
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        ew = self.half_w * c + self.half_h * s
        eh = self.half_w * s + self.half_h * c
        return (self.x - ew, self.y - eh, self.x + ew, self.y + eh)

    def contains_world(self, wx: float, wy: float) -> bool:
        """Check if a world-frame point is inside this obstacle's OBB."""
        dx = wx - self.x
        dy = wy - self.y
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        return abs(lx) <= self.half_w and abs(ly) <= self.half_h

    def nearest_point_to(self, wx: float, wy: float
                         ) -> Tuple[float, float]:
        """Return the nearest point on this obstacle's OBB to the world
        point (wx, wy), in world coordinates.

        For elongated obstacles (walls, tables) the nearest point lives
        on an edge of the bounding box, NOT at the centre — using it
        for direction-finding (e.g. orienting an MDA access point so
        the robot faces the closest part of a wall) gives perpendicular
        yaw to long surfaces, which is what users expect.

        Math: transform the query point into obstacle-local frame,
        clamp to ±half_w / ±half_h, then transform back.
        """
        dx = wx - self.x
        dy = wy - self.y
        ci, si = math.cos(-self.yaw), math.sin(-self.yaw)
        lx =  ci * dx - si * dy
        ly =  si * dx + ci * dy
        # Clamp into the OBB
        nlx = max(-self.half_w, min(self.half_w, lx))
        nly = max(-self.half_h, min(self.half_h, ly))
        # Back to world frame (forward rotation = +yaw)
        cf, sf = math.cos(self.yaw), math.sin(self.yaw)
        nx = self.x + cf * nlx - sf * nly
        ny = self.y + sf * nlx + cf * nly
        return (nx, ny)

    def distance_to(self, wx: float, wy: float) -> float:
        """Shortest distance from the world point (wx, wy) to this
        obstacle's OBB surface.  Returns 0 if the point is inside."""
        if self.contains_world(wx, wy):
            return 0.0
        nx, ny = self.nearest_point_to(wx, wy)
        return math.hypot(nx - wx, ny - wy)

    def circle_overlap(self, cx: float, cy: float, radius: float
                       ) -> Optional[Tuple[float, float, float]]:
        """
        Test overlap between a circle (robot) at (cx, cy) with given
        radius and this obstacle's OBB.

        Returns (push_x, push_y, overlap) in WORLD frame if overlapping,
        where (push_x, push_y) is the unit direction to push the circle
        OUT of the OBB, and overlap is the penetration depth.
        Returns None if no overlap.
        """
        # Transform circle centre into obstacle's local frame
        dx = cx - self.x
        dy = cy - self.y
        co = math.cos(-self.yaw)
        so = math.sin(-self.yaw)
        lx = co * dx - so * dy
        ly = so * dx + co * dy

        # Clamp to nearest point on the OBB surface
        nearest_lx = max(-self.half_w, min(self.half_w, lx))
        nearest_ly = max(-self.half_h, min(self.half_h, ly))

        # Distance from circle centre to nearest point
        dlx = lx - nearest_lx
        dly = ly - nearest_ly
        dist_sq = dlx * dlx + dly * dly

        if dist_sq >= radius * radius:
            return None   # no overlap

        dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
        overlap = radius - dist

        if dist > 1e-6:
            # Normal direction in local frame (away from OBB)
            nlx = dlx / dist
            nly = dly / dist
        else:
            # Circle centre is inside the OBB — push along shortest axis
            pen_x = self.half_w - abs(lx)
            pen_y = self.half_h - abs(ly)
            if pen_x < pen_y:
                nlx = 1.0 if lx >= 0 else -1.0
                nly = 0.0
                overlap = pen_x + radius
            else:
                nlx = 0.0
                nly = 1.0 if ly >= 0 else -1.0
                overlap = pen_y + radius

        # Rotate normal back to world frame
        co_fwd = math.cos(self.yaw)
        so_fwd = math.sin(self.yaw)
        wx = co_fwd * nlx - so_fwd * nly
        wy = so_fwd * nlx + co_fwd * nly

        return (wx, wy, overlap)

    def circle_overlap_casters(self, cx: float, cy: float, radius: float
                               ) -> Optional[Tuple[float, float, float]]:
        """
        Like circle_overlap but only tests against the 4 corner caster
        wheels (for HIGH/HEAVY trolleys where the robot can pass under
        the body).  Returns the worst (deepest) overlap among casters,
        or None if no overlap with any caster.
        """
        if not self.is_caster_trolley:
            return self.circle_overlap(cx, cy, radius)

        worst = None
        for (wx, wy, cr) in self.caster_circles():
            dx = cx - wx
            dy = cy - wy
            dist = math.hypot(dx, dy)
            min_dist = radius + cr
            if dist >= min_dist:
                continue
            overlap = min_dist - dist
            if dist > 1e-6:
                px, py = dx / dist, dy / dist
            else:
                px, py = 1.0, 0.0
            if worst is None or overlap > worst[2]:
                worst = (px, py, overlap)
        return worst

    def collision_overlap(self, cx: float, cy: float, radius: float
                          ) -> Optional[Tuple[float, float, float]]:
        """Unified collision check: uses caster-only for HIGH/HEAVY trolleys,
        full OBB for everything else."""
        if self.is_caster_trolley:
            return self.circle_overlap_casters(cx, cy, radius)
        return self.circle_overlap(cx, cy, radius)

    # ---- dynamic update (called every tick for DYNAMIC obstacles) --------

    def tick_dynamic(self, dt: float, world_bounds: Tuple[float,float,float,float]) -> None:
        """Advance dynamic obstacle state by one tick."""
        if self.kind == ObstacleKind.HUMAN:
            self._tick_human(dt, world_bounds)
        elif self.kind == ObstacleKind.SLIDING_DOOR:
            self._tick_sliding_door(dt)
        elif self.kind == ObstacleKind.PIVOT_DOOR:
            self._tick_pivot_door(dt)

    def _tick_human(self, dt: float, bounds: Tuple[float,float,float,float]) -> None:
        """Random walk: pick a nearby target, walk toward it at ~0.5 m/s,
        then pick a new target after reaching it or after a timeout."""
        speed = 0.5
        min_x, min_y, max_x, max_y = bounds

        self._walk_timer -= 1
        if self._walk_timer <= 0:
            # pick new random target within ±3 m, clamped to world bounds
            self._walk_target_x = max(min_x + 0.3, min(max_x - 0.3,
                self.x + random.uniform(-3.0, 3.0)))
            self._walk_target_y = max(min_y + 0.3, min(max_y - 0.3,
                self.y + random.uniform(-3.0, 3.0)))
            self._walk_timer = random.randint(60, 180)  # 2-6 sec at 30 fps

        dx = self._walk_target_x - self.x
        dy = self._walk_target_y - self.y
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            self._walk_timer = 0      # arrived, pick new target next tick
            return
        step = min(speed * dt, dist)
        self.x += (dx / dist) * step
        self.y += (dy / dist) * step
        self.yaw = math.atan2(dy, dx)

    def _tick_sliding_door(self, dt: float) -> None:
        """Proximity-based slide: open when _door_open, close when not.
        Slides full length along local X axis."""
        travel = self.half_w * 2        # slide by full door length
        speed  = 1.2                    # m/s (fast enough to feel responsive)
        target_offset = travel if self._door_open else 0.0
        diff = target_offset - self._slide_offset
        if abs(diff) < 0.005:
            self._slide_offset = target_offset
            return
        step = min(speed * dt, abs(diff))
        move = step if diff > 0 else -step
        self._slide_offset += move
        # Update position from closed position + offset along local X
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x = self._closed_x + c * self._slide_offset
        self.y = self._closed_y + s * self._slide_offset

    def _tick_pivot_door(self, dt: float) -> None:
        """Proximity-based pivot: swing to 90° when _door_open, back to 0°.
        Pivot/hinge at _closed_x/_closed_y (the -half_w end of the door).
        The door centre orbits around the hinge as it swings."""
        target_angle = math.pi / 2 if self._door_open else 0.0
        swing_speed = 1.5   # rad/s
        diff = target_angle - self._pivot_angle
        if abs(diff) < 0.01:
            self._pivot_angle = target_angle
        else:
            step = min(swing_speed * dt, abs(diff))
            self._pivot_angle += step if diff > 0 else -step
        # Update yaw
        self.yaw = self._pivot_base_yaw + self._pivot_angle
        # Recompute centre position: hinge + half_w along current yaw
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x = self._closed_x + c * self.half_w
        self.y = self._closed_y + s * self.half_w


# ============================================================================
#  Obstacle Manager
# ============================================================================

class ObstacleManager:
    """Owns the list of obstacles and handles spawn/despawn/drag/rendering."""

    def __init__(self) -> None:
        self.obstacles: Dict[int, Obstacle] = {}
        self.dragging_id: Optional[int] = None
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)

    # ---- spawn / despawn -----------------------------------------------

    def spawn(self, kind: ObstacleKind, x: float, y: float,
              yaw: float = 0.0, half_w: Optional[float] = None,
              half_h: Optional[float] = None) -> Obstacle:
        obs = Obstacle(kind=kind, x=x, y=y, yaw=yaw)
        if half_w is not None:
            obs._custom_half_w = half_w
        if half_h is not None:
            obs._custom_half_h = half_h
        if obs.kind in _TICKED:
            obs._walk_target_x = x
            obs._walk_target_y = y
            obs._walk_timer = random.randint(30, 90)
        # Doors: record closed position + base yaw
        if kind == ObstacleKind.SLIDING_DOOR:
            obs._closed_x = x
            obs._closed_y = y
        elif kind == ObstacleKind.PIVOT_DOOR:
            # _closed_x/y = hinge (click position, the -half_w end)
            obs._closed_x = x
            obs._closed_y = y
            obs._pivot_base_yaw = yaw
            # Offset centre from hinge by half_w along yaw
            c, s = math.cos(yaw), math.sin(yaw)
            obs.x = x + c * obs.half_w
            obs.y = y + s * obs.half_w
        self.obstacles[obs.uid] = obs
        return obs

    def despawn(self, uid: int) -> bool:
        if uid in self.obstacles:
            del self.obstacles[uid]
            if self.dragging_id == uid:
                self.dragging_id = None
            return True
        return False

    def obstacle_at(self, wx: float, wy: float) -> Optional[Obstacle]:
        """Return the top-most obstacle containing world point (wx, wy)."""
        for obs in reversed(list(self.obstacles.values())):
            if obs.contains_world(wx, wy):
                return obs
        return None

    # ---- drag ----------------------------------------------------------

    def start_drag(self, uid: int, wx: float, wy: float) -> None:
        obs = self.obstacles.get(uid)
        if obs is None:
            return
        self.dragging_id = uid
        self._drag_offset = (obs.x - wx, obs.y - wy)

    def update_drag(self, wx: float, wy: float) -> None:
        if self.dragging_id is None:
            return
        obs = self.obstacles.get(self.dragging_id)
        if obs is None:
            self.dragging_id = None
            return
        new_x = wx + self._drag_offset[0]
        new_y = wy + self._drag_offset[1]
        dx = new_x - obs.x
        dy = new_y - obs.y
        obs.x = new_x
        obs.y = new_y
        # Move anchor points for doors so tick doesn't snap them back
        # and the proximity activation zone follows the door.
        if obs.kind == ObstacleKind.SLIDING_DOOR:
            obs._closed_x += dx
            obs._closed_y += dy
        elif obs.kind == ObstacleKind.PIVOT_DOOR:
            obs._closed_x += dx
            obs._closed_y += dy
        # Move walk target for humans so they don't wander back
        if obs.kind == ObstacleKind.HUMAN:
            obs._walk_target_x += dx
            obs._walk_target_y += dy

    def end_drag(self) -> None:
        self.dragging_id = None

    # ---- door proximity detection ---------------------------------------
    # Proximity range: 1.5 × robot_length (0.70m × 1.5 = 1.05m)
    DOOR_PROXIMITY_M: float = 1.05
    # Pivot door back/swing-side proximity (larger so the door opens
    # earlier when approached from the side it swings into).
    # The swing side is +local_Y in the closed reference frame.
    PIVOT_BACK_PROXIMITY_M: float = 2.10   # 3 × robot_length

    def update_door_proximity(
            self,
            entity_positions: List[Tuple[float, float]],
    ) -> None:
        """Check proximity of entities (robots, humans) to doors.
        Sets _door_open = True if any entity is within proximity of the
        door's long edge at its closed position, else False.

        For pivot doors the detection zone is asymmetric: the back/swing
        side (+local_Y) uses PIVOT_BACK_PROXIMITY_M while the front side
        (-local_Y) uses the normal DOOR_PROXIMITY_M."""
        prox = self.DOOR_PROXIMITY_M
        back_prox = self.PIVOT_BACK_PROXIMITY_M
        for obs in self.obstacles.values():
            if obs.kind not in (ObstacleKind.SLIDING_DOOR,
                                ObstacleKind.PIVOT_DOOR):
                continue
            # Proximity zone: rectangle extending ±prox from the long
            # edges of the door at its CLOSED position.
            # Local frame: long axis = X (±half_w), short axis = Y (±half_h)
            # (using the closed position/yaw for the reference frame)
            is_pivot = obs.kind == ObstacleKind.PIVOT_DOOR
            if is_pivot:
                base_yaw = obs._pivot_base_yaw
                # Proximity centred on door's closed midpoint (hinge + half_w)
                pc = math.cos(base_yaw)
                ps = math.sin(base_yaw)
                cx = obs._closed_x + pc * obs.half_w
                cy = obs._closed_y + ps * obs.half_w
            else:
                base_yaw = obs.yaw  # sliding door yaw doesn't change
                cx, cy = obs._closed_x, obs._closed_y
            cos_y = math.cos(base_yaw)
            sin_y = math.sin(base_yaw)
            hw = obs.half_w
            hh = obs.half_h
            opened = False
            for (ex, ey) in entity_positions:
                # Transform to door's local frame at closed position
                dx = ex - cx
                dy = ey - cy
                lx =  cos_y * dx + sin_y * dy
                ly = -sin_y * dx + cos_y * dy
                # Check if within the proximity zone
                if abs(lx) < hw + prox:
                    if is_pivot:
                        # Asymmetric: back/swing side (+Y) has larger range
                        if ly > 0:
                            y_ok = ly < hh + back_prox
                        else:
                            y_ok = -ly < hh + prox
                    else:
                        y_ok = abs(ly) < hh + prox
                    if y_ok:
                        opened = True
                        break
            obs._door_open = opened

    # ---- tick all dynamic obstacles ------------------------------------

    def tick(self, dt: float, world_bounds: Tuple[float,float,float,float]) -> None:
        for obs in self.obstacles.values():
            if obs.kind in _TICKED and obs.uid != self.dragging_id:
                obs.tick_dynamic(dt, world_bounds)

    # ---- MDA mount / unmount (Phase E / Pressing 5) --------------------

    def find_mda_for_robot(self, robot_id: int) -> Optional[Obstacle]:
        """Return the MDA module mounted on the given robot, or None."""
        for obs in self.obstacles.values():
            if obs.is_mda_module and obs.host_robot_id == robot_id:
                return obs
        return None

    def find_unmounted_mda_near(self, x: float, y: float,
                                max_range: float = MDA_MOUNT_RANGE_M
                                ) -> Optional[Obstacle]:
        """Closest unmounted MDA module within ``max_range`` of (x, y).

        Returns None if no qualifying module exists.  Used by demo.py's
        '0' key to find a candidate to mount onto the active robot.
        """
        best: Optional[Obstacle] = None
        best_d2 = max_range * max_range
        for obs in self.obstacles.values():
            if not obs.is_mda_module or obs.is_mounted:
                continue
            d2 = (obs.x - x) ** 2 + (obs.y - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = obs
        return best

    def mount_mda(self, mda_uid: int, host_robot_id: int,
                  host_xy: Tuple[float, float],
                  host_yaw: float = 0.0) -> bool:
        """Attach the MDA module ``mda_uid`` to ``host_robot_id``.

        Snaps the module's pose to the host and clears any active drag.
        Enforces the strategy-doc rule "one MDA per robot" by refusing
        to mount if the host already has one (returns False).  Caller
        is responsible for the "one trolley-bearer at a time" check
        (trolley state lives in Configurer.SimBus, not here).
        """
        obs = self.obstacles.get(mda_uid)
        if obs is None or not obs.is_mda_module or obs.is_mounted:
            return False
        if self.find_mda_for_robot(host_robot_id) is not None:
            return False                 # one MDA per robot
        obs.host_robot_id = host_robot_id
        obs.x, obs.y = host_xy
        obs.yaw = host_yaw
        if self.dragging_id == mda_uid:
            self.dragging_id = None
        return True

    def unmount_mda(self, mda_uid: int) -> bool:
        """Detach a mounted MDA module — leaves it as a free obstacle
        at the host's last known pose.  Returns True on success."""
        obs = self.obstacles.get(mda_uid)
        if obs is None or not obs.is_mounted:
            return False
        obs.host_robot_id = None
        return True

    def sync_mounted_mdas(self,
                          robot_poses: Dict[int, Tuple[float, float, float]]
                          ) -> None:
        """Snap every mounted MDA's pose to its host robot.

        Call this once per tick from demo.py BEFORE rendering so that
        the MDA visual stays glued to its host as the robot drives.
        ``robot_poses`` maps robot_id → (x, y, yaw).  Mounts whose
        host has been removed (key missing) get auto-unmounted.
        """
        for obs in self.obstacles.values():
            if not obs.is_mounted:
                continue
            host = robot_poses.get(obs.host_robot_id)
            if host is None:
                obs.host_robot_id = None         # host gone → detach
                continue
            obs.x, obs.y, obs.yaw = host

    def has_mda_mounted(self, robot_id: int) -> bool:
        """Convenience predicate for Mode 5 precondition checks."""
        return self.find_mda_for_robot(robot_id) is not None

    def any_mda_mounted(self) -> bool:
        """True if any robot in the scene has an MDA mounted.

        Pressing '5' uses this as the gate to enter Mode 5 — the
        strategy doc requires "MDA module mounted on a robot" before
        the GBNN+H planner becomes active.
        """
        return any(o.is_mounted for o in self.obstacles.values())

    # ---- occupancy grid ------------------------------------------------

    def build_occupancy_grid(
        self,
        world_bounds: Tuple[float, float, float, float],
        cell_size: float = 0.20,
        inflate_radius: float = 0.45,
        robot_positions: Optional[List[Tuple[float, float, float]]] = None,
        exclude_rid: Optional[int] = None,
    ) -> "OccupancyGrid":
        """Build a 2D boolean grid where True = occupied/inflated.

        Parameters
        ----------
        robot_positions : list of (x, y, occupancy_radius) for each robot
            If provided, other robots are marked as obstacles so the
            pathfinder routes around them.
        exclude_rid : robot id to exclude (the navigating robot itself).
            Index into *robot_positions* (0-based = rid-1 if 1-indexed).
        """
        min_x, min_y, max_x, max_y = world_bounds
        cols = max(1, int(math.ceil((max_x - min_x) / cell_size)))
        rows = max(1, int(math.ceil((max_y - min_y) / cell_size)))
        grid = OccupancyGrid(
            min_x=min_x, min_y=min_y,
            cell_size=cell_size,
            cols=cols, rows=rows,
        )

        def _mark_rect(ax0, ay0, ax1, ay1):
            c0 = max(0, int((ax0 - min_x) / cell_size))
            r0 = max(0, int((ay0 - min_y) / cell_size))
            c1 = min(cols - 1, int((ax1 - min_x) / cell_size))
            r1 = min(rows - 1, int((ay1 - min_y) / cell_size))
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    grid.data[r][c] = True

        # Mark cells occupied by each obstacle (inflated by robot radius)
        for obs in self.obstacles.values():
            if obs.is_mounted:
                # Mounted MDA modules are NOT independent obstacles — the
                # host robot's footprint inflates to MDA_ARM_RADIUS via the
                # robot_positions list passed in by the caller.  Skip here
                # to avoid double-blocking the host's own cell.
                continue
            if obs.is_caster_trolley:
                # Only mark caster wheel circles, not the full body
                for (wcx, wcy, cr) in obs.caster_circles():
                    total = cr + inflate_radius
                    _mark_rect(wcx - total, wcy - total,
                               wcx + total, wcy + total)
            else:
                ax0, ay0, ax1, ay1 = obs.aabb()
                _mark_rect(ax0 - inflate_radius, ay0 - inflate_radius,
                           ax1 + inflate_radius, ay1 + inflate_radius)

        # Mark cells occupied by other robots
        if robot_positions:
            for i, (rx, ry, r_occ) in enumerate(robot_positions):
                if exclude_rid is not None and i == exclude_rid:
                    continue
                # Treat each robot as a circle → inflate by planner's
                # inflate_radius to keep clearance
                total_r = r_occ + inflate_radius
                _mark_rect(rx - total_r, ry - total_r,
                           rx + total_r, ry + total_r)

        return grid

    # ---- rendering (called from TeleopSim._render) ---------------------

    def draw_all(self, screen, world_to_screen, metres_to_px, font) -> None:
        """Draw every obstacle onto the pygame surface.

        Mounted MDA modules are skipped here and drawn separately by
        ``draw_mounted_mda`` AFTER robots so they render on top of
        their host (z+1, per strategy doc §312).
        """
        for obs in self.obstacles.values():
            if obs.is_mounted:
                continue
            _draw_obstacle(screen, obs, world_to_screen, metres_to_px, font)

    def draw_non_caster(self, screen, world_to_screen, metres_to_px,
                        font) -> None:
        """Draw only non-caster obstacles (everything except HIGH/HEAVY
        trolleys).  Mounted MDAs are also excluded — see ``draw_all``."""
        for obs in self.obstacles.values():
            if obs.kind in _CASTER_TROLLEYS:
                continue
            if obs.is_mounted:
                continue
            _draw_obstacle(screen, obs, world_to_screen, metres_to_px, font)

    def draw_caster_only(self, screen, world_to_screen, metres_to_px,
                         font) -> None:
        """Draw only caster trolleys (HIGH/HEAVY) — called later in z-order."""
        for obs in self.obstacles.values():
            if obs.kind in _CASTER_TROLLEYS:
                _draw_obstacle(screen, obs, world_to_screen, metres_to_px, font)

    def draw_mounted_mda(self, screen, world_to_screen, metres_to_px,
                         font) -> None:
        """Draw mounted MDA modules — called AFTER robots in the render
        pipeline so they appear above their host (z+1)."""
        for obs in self.obstacles.values():
            if obs.is_mounted:
                _draw_obstacle(screen, obs, world_to_screen, metres_to_px, font)


# ============================================================================
#  Occupancy Grid + Pathfinding
# ============================================================================

@dataclass
class OccupancyGrid:
    min_x:     float
    min_y:     float
    cell_size: float
    cols:      int
    rows:      int
    data:      List[List[bool]] = field(default=None)

    def __post_init__(self):
        if self.data is None:
            self.data = [[False] * self.cols for _ in range(self.rows)]

    def world_to_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        c = int((wx - self.min_x) / self.cell_size)
        r = int((wy - self.min_y) / self.cell_size)
        return (max(0, min(self.cols - 1, c)),
                max(0, min(self.rows - 1, r)))

    def cell_to_world(self, c: int, r: int) -> Tuple[float, float]:
        wx = self.min_x + (c + 0.5) * self.cell_size
        wy = self.min_y + (r + 0.5) * self.cell_size
        return wx, wy

    def is_free(self, c: int, r: int) -> bool:
        if 0 <= c < self.cols and 0 <= r < self.rows:
            return not self.data[r][c]
        return False

    def neighbours(self, c: int, r: int) -> List[Tuple[int, int, float]]:
        """Return (col, row, cost) for 8-connected neighbours."""
        result = []
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                nc, nr = c + dc, r + dr
                if self.is_free(nc, nr):
                    cost = 1.414 if (dc != 0 and dr != 0) else 1.0
                    result.append((nc, nr, cost))
        return result

    def nearest_free_cell(self, c: int, r: int,
                          max_radius: int = 10) -> Optional[Tuple[int, int]]:
        """BFS outward from (c, r) to find the nearest free cell.
        Returns (col, row) or None if nothing free within max_radius."""
        if self.is_free(c, r):
            return (c, r)
        from collections import deque
        visited = {(c, r)}
        queue = deque([(c, r, 0)])
        while queue:
            cc, cr, dist = queue.popleft()
            if dist > max_radius:
                break
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = cc + dc, cr + dr
                    if (nc, nr) in visited:
                        continue
                    visited.add((nc, nr))
                    if 0 <= nc < self.cols and 0 <= nr < self.rows:
                        if self.is_free(nc, nr):
                            return (nc, nr)
                        queue.append((nc, nr, dist + 1))
        return None


# ---- A* pathfinder -------------------------------------------------------

import heapq

def pathfind_astar(
    grid: OccupancyGrid,
    start_world: Tuple[float, float],
    goal_world:  Tuple[float, float],
) -> Optional[List[Tuple[float, float]]]:
    """A* on the occupancy grid. Returns list of world-frame waypoints
    or None if no path exists.  If start or goal cell is blocked
    (e.g. robot inside inflated zone), the nearest free cell is used."""
    sc, sr = grid.world_to_cell(*start_world)
    gc, gr = grid.world_to_cell(*goal_world)

    # If start or goal is inside an inflated obstacle, snap to nearest free
    if not grid.is_free(sc, sr):
        free = grid.nearest_free_cell(sc, sr)
        if free is None:
            return None
        sc, sr = free
    if not grid.is_free(gc, gr):
        free = grid.nearest_free_cell(gc, gr)
        if free is None:
            return None
        gc, gr = free

    def heuristic(c: int, r: int) -> float:
        return math.hypot(c - gc, r - gr)

    open_set: List[Tuple[float, int, int]] = []
    heapq.heappush(open_set, (heuristic(sc, sr), sc, sr))
    came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {(sc, sr): None}
    g_score: Dict[Tuple[int,int], float] = {(sc, sr): 0.0}

    while open_set:
        _, cc, cr = heapq.heappop(open_set)
        if cc == gc and cr == gr:
            # reconstruct
            path_cells = []
            node = (gc, gr)
            while node is not None:
                path_cells.append(node)
                node = came_from[node]
            path_cells.reverse()
            return [grid.cell_to_world(c, r) for c, r in path_cells]

        for nc, nr, cost in grid.neighbours(cc, cr):
            tentative = g_score[(cc, cr)] + cost
            if (nc, nr) not in g_score or tentative < g_score[(nc, nr)]:
                g_score[(nc, nr)] = tentative
                f = tentative + heuristic(nc, nr)
                came_from[(nc, nr)] = (cc, cr)
                heapq.heappush(open_set, (f, nc, nr))

    return None


# ---- Dijkstra pathfinder -------------------------------------------------

def pathfind_dijkstra(
    grid: OccupancyGrid,
    start_world: Tuple[float, float],
    goal_world:  Tuple[float, float],
) -> Optional[List[Tuple[float, float]]]:
    """Dijkstra on the occupancy grid. Returns list of world-frame waypoints
    or None if no path exists.  If start or goal cell is blocked,
    the nearest free cell is used."""
    sc, sr = grid.world_to_cell(*start_world)
    gc, gr = grid.world_to_cell(*goal_world)

    # If start or goal is inside an inflated obstacle, snap to nearest free
    if not grid.is_free(sc, sr):
        free = grid.nearest_free_cell(sc, sr)
        if free is None:
            return None
        sc, sr = free
    if not grid.is_free(gc, gr):
        free = grid.nearest_free_cell(gc, gr)
        if free is None:
            return None
        gc, gr = free

    open_set: List[Tuple[float, int, int]] = []
    heapq.heappush(open_set, (0.0, sc, sr))
    came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {(sc, sr): None}
    g_score: Dict[Tuple[int,int], float] = {(sc, sr): 0.0}

    while open_set:
        dist, cc, cr = heapq.heappop(open_set)
        if cc == gc and cr == gr:
            path_cells = []
            node = (gc, gr)
            while node is not None:
                path_cells.append(node)
                node = came_from[node]
            path_cells.reverse()
            return [grid.cell_to_world(c, r) for c, r in path_cells]

        if dist > g_score.get((cc, cr), math.inf):
            continue

        for nc, nr, cost in grid.neighbours(cc, cr):
            tentative = dist + cost
            if (nc, nr) not in g_score or tentative < g_score[(nc, nr)]:
                g_score[(nc, nr)] = tentative
                came_from[(nc, nr)] = (cc, cr)
                heapq.heappush(open_set, (tentative, nc, nr))

    return None


# ---- path smoothing --------------------------------------------------------

def smooth_path(
    grid: OccupancyGrid,
    path: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Line-of-sight path simplification.

    Walks through the path and skips intermediate waypoints whenever
    a straight line between two non-adjacent waypoints is collision-free
    on the grid.  Produces far fewer waypoints and much smoother motion.
    """
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Try to skip as far ahead as possible from i
        best = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if _line_of_sight(grid, path[i], path[j]):
                best = j
                break
        smoothed.append(path[best])
        i = best
    return smoothed


def _line_of_sight(
    grid: OccupancyGrid,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
) -> bool:
    """Check that a straight line between two world points passes only
    through free cells (Bresenham-style walk on the grid)."""
    c0, r0 = grid.world_to_cell(*p0)
    c1, r1 = grid.world_to_cell(*p1)
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c1 > c0 else -1
    sr = 1 if r1 > r0 else -1
    err = dc - dr
    cc, cr = c0, r0
    while True:
        if not grid.is_free(cc, cr):
            return False
        if cc == c1 and cr == r1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            cc += sc
        if e2 < dc:
            err += dc
            cr += sr
    return True


# ============================================================================
#  Rendering helpers (simple shapes with recognisable features)
# ============================================================================

def _draw_obstacle(screen, obs: Obstacle, w2s, m2px, font) -> None:
    """Draw a single obstacle with recognisable visual features."""
    import pygame

    fill, outline = OBSTACLE_COLOURS[obs.kind]
    cx, cy = w2s(obs.x, obs.y)
    hw_px = max(4, m2px(obs.half_w))
    hh_px = max(4, m2px(obs.half_h))

    c, s = math.cos(obs.yaw), math.sin(obs.yaw)

    def _rotated_rect(hw, hh, ox=0, oy=0):
        """Return 4 screen-space corners of a rotated rect centred at obs."""
        local = [(-hw + ox, -hh + oy), (hw + ox, -hh + oy),
                 (hw + ox,  hh + oy), (-hw + ox,  hh + oy)]
        pts = []
        for lx, ly in local:
            rx =  c * lx - s * (-ly)
            ry =  s * lx + c * (-ly)
            pts.append((cx + rx, cy - ry))
        return pts

    # ---- TABLE: brown rectangle + 4 small corner leg circles -----------
    if obs.kind == ObstacleKind.TABLE:
        body = _rotated_rect(hw_px, hh_px)
        pygame.draw.polygon(screen, fill, body)
        pygame.draw.polygon(screen, outline, body, width=2)
        # legs at each corner (small circles)
        leg_r = max(2, hw_px // 6)
        for lx_f, ly_f in [(-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8)]:
            lx = lx_f * hw_px
            ly = ly_f * hh_px
            rx =  c * lx - s * (-ly)
            ry =  s * lx + c * (-ly)
            pygame.draw.circle(screen, outline, (int(cx + rx), int(cy - ry)), leg_r)

    # ---- CHAIR: small square + backrest bar on one edge ----------------
    elif obs.kind == ObstacleKind.CHAIR:
        body = _rotated_rect(hw_px, hh_px)
        pygame.draw.polygon(screen, fill, body)
        pygame.draw.polygon(screen, outline, body, width=2)
        # backrest: thick bar along the back edge (-hh side)
        bar_hw = hw_px
        bar_hh = max(2, hh_px // 4)
        bar = _rotated_rect(bar_hw, bar_hh, ox=0, oy=-hh_px + bar_hh)
        pygame.draw.polygon(screen, outline, bar)

    # ---- HIGH TROLLEY: semi-transparent body + 4 caster circles --------
    elif obs.kind == ObstacleKind.HIGH_TROLLEY:
        body = _rotated_rect(hw_px, hh_px)
        # Lighter fill to indicate robot can pass under
        body_col = tuple(min(255, ch + 40) for ch in fill)
        pygame.draw.polygon(screen, body_col, body)
        pygame.draw.polygon(screen, outline, body, width=1)
        # Dashed cross to show "passable underneath"
        pygame.draw.line(screen, outline, (cx - hw_px//2, cy), (cx + hw_px//2, cy), 1)
        pygame.draw.line(screen, outline, (cx, cy - hh_px//2), (cx, cy + hh_px//2), 1)
        # 4 caster wheel circles at corners
        caster_px = max(3, m2px(CASTER_RADIUS))
        cr_inset = CASTER_RADIUS
        for lx_m, ly_m in [(-obs.half_w + cr_inset, -obs.half_h + cr_inset),
                            ( obs.half_w - cr_inset, -obs.half_h + cr_inset),
                            ( obs.half_w - cr_inset,  obs.half_h - cr_inset),
                            (-obs.half_w + cr_inset,  obs.half_h - cr_inset)]:
            px_x = m2px(lx_m)
            px_y = m2px(ly_m)
            rx_s =  c * px_x - s * (-px_y)
            ry_s =  s * px_x + c * (-px_y)
            pygame.draw.circle(screen, (60, 60, 70),
                               (int(cx + rx_s), int(cy - ry_s)), caster_px)
            pygame.draw.circle(screen, (40, 40, 50),
                               (int(cx + rx_s), int(cy - ry_s)), caster_px, width=1)

    # ---- LOW TROLLEY: solid rect + 4 small wheels (blocks passage) -----
    elif obs.kind == ObstacleKind.LOW_TROLLEY:
        body = _rotated_rect(hw_px, hh_px)
        pygame.draw.polygon(screen, fill, body)
        pygame.draw.polygon(screen, outline, body, width=2)
        # 4 wheels at corners
        wheel_r = max(2, min(hw_px, hh_px) // 5)
        for lx_f, ly_f in [(-0.7, -0.7), (0.7, -0.7), (0.7, 0.7), (-0.7, 0.7)]:
            lx = lx_f * hw_px
            ly = ly_f * hh_px
            rx = c * lx - s * (-ly)
            ry = s * lx + c * (-ly)
            pygame.draw.circle(screen, (50, 50, 60),
                               (int(cx + rx), int(cy - ry)), wheel_r)

    # ---- HEAVY TROLLEY: like HIGH but darker, with 4 caster circles ----
    elif obs.kind == ObstacleKind.HEAVY_TROLLEY:
        body = _rotated_rect(hw_px, hh_px)
        body_col = tuple(min(255, ch + 30) for ch in fill)
        pygame.draw.polygon(screen, body_col, body)
        pygame.draw.polygon(screen, outline, body, width=2)
        # Weight indicator: double cross
        pygame.draw.line(screen, outline, (cx - hw_px//2, cy - 2), (cx + hw_px//2, cy - 2), 1)
        pygame.draw.line(screen, outline, (cx - hw_px//2, cy + 2), (cx + hw_px//2, cy + 2), 1)
        pygame.draw.line(screen, outline, (cx - 2, cy - hh_px//2), (cx - 2, cy + hh_px//2), 1)
        pygame.draw.line(screen, outline, (cx + 2, cy - hh_px//2), (cx + 2, cy + hh_px//2), 1)
        # 4 caster wheel circles at corners
        caster_px = max(3, m2px(CASTER_RADIUS))
        cr_inset = CASTER_RADIUS
        for lx_m, ly_m in [(-obs.half_w + cr_inset, -obs.half_h + cr_inset),
                            ( obs.half_w - cr_inset, -obs.half_h + cr_inset),
                            ( obs.half_w - cr_inset,  obs.half_h - cr_inset),
                            (-obs.half_w + cr_inset,  obs.half_h - cr_inset)]:
            px_x = m2px(lx_m)
            px_y = m2px(ly_m)
            rx_s =  c * px_x - s * (-px_y)
            ry_s =  s * px_x + c * (-px_y)
            pygame.draw.circle(screen, (50, 50, 60),
                               (int(cx + rx_s), int(cy - ry_s)), caster_px)
            pygame.draw.circle(screen, (35, 35, 45),
                               (int(cx + rx_s), int(cy - ry_s)), caster_px, width=1)

    # ---- WALL: long thin grey rectangle with brick-like hatching ----------
    elif obs.kind == ObstacleKind.WALL:
        body = _rotated_rect(hw_px, hh_px)
        pygame.draw.polygon(screen, fill, body)
        pygame.draw.polygon(screen, outline, body, width=2)
        # centre line for brick pattern (along the longer axis)
        if hw_px >= hh_px:
            mid_l = _rotated_rect(hw_px, 0, ox=0, oy=0)
        else:
            mid_l = _rotated_rect(0, hh_px, ox=0, oy=0)
        if len(mid_l) >= 2:
            pygame.draw.line(screen, outline, mid_l[0], mid_l[1], 1)

    # ---- PILLAR: filled circle with cross pattern ----------------------
    elif obs.kind == ObstacleKind.PILLAR:
        radius = max(4, hw_px)
        pygame.draw.circle(screen, fill, (cx, cy), radius)
        pygame.draw.circle(screen, outline, (cx, cy), radius, width=2)
        # cross
        arm = radius - 2
        pygame.draw.line(screen, outline, (cx - arm, cy), (cx + arm, cy), 1)
        pygame.draw.line(screen, outline, (cx, cy - arm), (cx, cy + arm), 1)

    # ---- SLIDING DOOR: thin blue rect with arrows ←→ ------------------
    elif obs.kind == ObstacleKind.SLIDING_DOOR:
        body = _rotated_rect(hw_px, hh_px)
        pygame.draw.polygon(screen, fill, body)
        pygame.draw.polygon(screen, outline, body, width=2)
        # small arrow hints
        arr = max(3, hw_px // 4)
        pygame.draw.line(screen, (200, 220, 240),
                         (cx - arr, cy), (cx + arr, cy), 2)
        # arrowheads
        pygame.draw.line(screen, (200, 220, 240),
                         (cx - arr, cy), (cx - arr + 3, cy - 3), 2)
        pygame.draw.line(screen, (200, 220, 240),
                         (cx + arr, cy), (cx + arr - 3, cy - 3), 2)

    # ---- PIVOT DOOR: thin green rect with arc indicator ----------------
    elif obs.kind == ObstacleKind.PIVOT_DOOR:
        body = _rotated_rect(hw_px, hh_px)
        pygame.draw.polygon(screen, fill, body)
        pygame.draw.polygon(screen, outline, body, width=2)
        # small arc at pivot end
        arc_r = max(4, hw_px // 3)
        pygame.draw.arc(screen, (150, 200, 170),
                        (cx - hw_px - arc_r, cy - arc_r,
                         arc_r * 2, arc_r * 2),
                        -0.3, 1.8, 2)

    # ---- MDA MODULE: hexagonal base + two arm stubs ---------------------
    # Distinct from robots and trolleys.  When mounted, drawn at z+1 above
    # the host (see ObstacleManager.draw_mounted_mda) and tinted with a
    # white outline to signal "attached".
    elif obs.kind == ObstacleKind.MDA_MODULE:
        radius = max(5, max(hw_px, hh_px))
        # Hexagonal base — six-point regular polygon, flat on top
        hex_pts = []
        for k in range(6):
            ang = obs.yaw + math.pi / 6 + k * (math.pi / 3)
            hx = cx + radius * math.cos(ang)
            hy = cy - radius * math.sin(ang)
            hex_pts.append((hx, hy))
        body_col = fill if not obs.is_mounted else \
                   tuple(min(255, ch + 25) for ch in fill)
        pygame.draw.polygon(screen, body_col, hex_pts)
        out_col = (240, 240, 240) if obs.is_mounted else outline
        out_w = 3 if obs.is_mounted else 2
        pygame.draw.polygon(screen, out_col, hex_pts, width=out_w)
        # Two arm stubs — small rectangles sticking out the ±perpendicular
        # to obs.yaw, symbolising the dual-arm payload
        arm_len = max(4, radius * 3 // 4)
        arm_w   = max(2, radius // 4)
        for sgn in (+1, -1):
            ang = obs.yaw + sgn * math.pi / 2
            ax = math.cos(ang)
            ay = math.sin(ang)
            # base of stub
            bx = cx + ax * radius * 0.5
            by = cy - ay * radius * 0.5
            tx = cx + ax * (radius * 0.5 + arm_len)
            ty = cy - ay * (radius * 0.5 + arm_len)
            pygame.draw.line(screen, outline,
                             (int(bx), int(by)), (int(tx), int(ty)),
                             arm_w)
            pygame.draw.circle(screen, outline,
                               (int(tx), int(ty)),
                               max(2, arm_w))
        # Heading line — short segment along obs.yaw to show arm-front
        fx = cx + int(math.cos(obs.yaw) * radius * 0.7)
        fy = cy - int(math.sin(obs.yaw) * radius * 0.7)
        pygame.draw.line(screen, (60, 60, 60), (cx, cy), (fx, fy), 2)

    # ---- HUMAN: circle (head) + triangle body --------------------------
    elif obs.kind == ObstacleKind.HUMAN:
        radius = max(4, hw_px)
        # body triangle
        body_pts = [
            (cx, cy + radius * 2),     # bottom
            (cx - radius, cy + radius * 2 + radius * 2),
            (cx + radius, cy + radius * 2 + radius * 2),
        ]
        # simpler: just a filled circle for head + smaller circle for body
        body_r = int(radius * 1.2)
        head_r = max(3, int(radius * 0.7))
        pygame.draw.circle(screen, fill, (cx, cy + head_r), body_r)         # body
        pygame.draw.circle(screen, outline, (cx, cy + head_r), body_r, 2)
        pygame.draw.circle(screen, (240, 190, 150), (cx, cy - head_r), head_r)  # head
        pygame.draw.circle(screen, outline, (cx, cy - head_r), head_r, 1)
        # direction indicator (small line from centre)
        fx = cx + int(math.cos(obs.yaw) * radius * 1.5)
        fy = cy - int(math.sin(obs.yaw) * radius * 1.5)
        pygame.draw.line(screen, (255, 100, 100), (cx, cy), (fx, fy), 2)

    # ---- label (small text above) --------------------------------------
    lbl_surf = font.render(obs.label, True, (200, 200, 210))
    lbl_pos = (cx - lbl_surf.get_width() // 2,
               cy - max(hw_px, hh_px) - 16)
    screen.blit(lbl_surf, lbl_pos)