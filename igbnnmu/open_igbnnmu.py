#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
open_igbnnmu.py  —  IGBNN-mu: Complete Coverage Path Planning for
                    Scalable Inter-Reconfigurable Robots
===============================================================================
Paper : "IGBNN-mu: Complete Coverage Path Planning for Scalable
         Inter-Reconfigurable Robots"
        IEEE Transactions on Systems, Man, and Cybernetics: Systems
Thesis: Ch.5 — Towards Computationally Scalable Inter-Reconfigurable Robots
Author: Ash Wan Yaw Sang  |  SUTD 2025  |  ROAR Lab

Filename note
-------------
The module file is named ``open_igbnnmu.py`` (lowercase, no '-' or 'μ') so it
imports cleanly via::

    from igbnnmu.open_igbnnmu import IGBNN_mu

— matching how ``open_configurer.py``, ``open_interstar.py`` and
``open_gbnnh.py`` are loaded by ``demo.py``.  Class names, docstrings and
paper citations retain the original "IGBNN-mu" notation.

Algorithm overview
------------------
Glasius Bioinspired Neural Network (GBNN) complete coverage, extended for
robots whose footprint changes as they fuse and split mid-sweep.  Four
components, per Section III:

  1. Inter-reconfiguration model (III-C)  — a robot adjacent to a lower-indexed
     robot follows it; the pair plans as one unit, so per-iteration waypoint
     computations fall as the team combines.        _xfm(), Algorithm 3
  2. Neuron skipping (III-D)              — a formation fused k-deep on an axis
     steps k cells along that axis at once.               _nav_nbrs()
  3. Minigraphs (III-E)                   — the map is partitioned into square
     tiles swept in serpentine order, each with its own local IGBNN run and an
     A* transit between them.               _plan_partition(), Algorithm 4
  4. Reduced decision space (III-C)       — only morphology-consistent
     neighbours are scored.                       _nav_nbrs(), _next_wp()

Measured gain: coverage effort per robot FALLS as the team grows (O(log n)^-1),
the inverse of standard multi-robot GBNN's O(n).

State vs activity
-----------------
The neural field and the coverage bookkeeping are kept in TWO separate arrays.
They are never the same array.

    self.state   (H, V)  coverage state — ground truth, monotone 1.0 -> 0.0
    self._act    (h, v)  neural activity over the CURRENT minigraph

The external bias I_ij (Eqn 5) is defined by a cell's STATE, not by its
activity value.  Sharing one array lets a visited cell whose weighted
neighbour input reaches s >= 1 be clamped by f() to exactly 1.0, become
indistinguishable from an unvisited cell, be re-assigned I_ij = +E, and be
re-attracted forever — so the inner loop never reports complete.  Measured on
a 30x30 map at 20% obstacle density: 12938 steps and 2 uncovered cells, versus
420 steps and full coverage once the arrays are separated.

Grid encoding
-------------
    -1.0 = obstacle / occupied
     1.0 = unvisited free cell
     0.0 = visited free cell  (accepted on input; useful for resume)

Robot state
-----------
S_i as defined in Section IV-C, laid out as a 9-element list:

    [x, y, z, dx, dy, alpha, beta, S_v, S_h]

    x, y        position — LOCAL to the current minigraph during the inner
                loop, GLOBAL during transit
    z           layer index (always 0 for 2D maps)
    dx, dy      host's last motion vector, followed by attached robots
    alpha       own_id of the host being followed; alpha == beta -> independent
    beta        permanent robot index, 0 ... n_robots-1
    S_v         own_ids fused vertically
    S_h         own_ids fused horizontally

Minigraph construction (Section IV-C)
-------------------------------------
Square partitions, minimal padding, Rule 2 (the tile edge must seat the team):

    side = max(rows, cols)
    u    = max( ceil(sqrt(side)), n )   # tile edge
    k    = ceil(side / u)               # minigraphs per axis
    L    = k * u                        # both axes padded to this side

Padded cells are marked occupied, fully blocked minigraphs are skipped, and a
minigraph with fewer free cells than robots seats only as many as it can — the
remainder idle until the next one.

Parameters (Table I)
--------------------
    E = 100     visited-node reinforcement constant (Eqn 5)
    q = 2       weight decay in w_ij = exp(-q||i-j||^2) (Eqn 4)
    a = 0.7     activation scaling in the linear regime (Eqn 2)
    R = 2       receptive-field radius (Eqn 4)
    lambda = 8  maximum number of neighbours (Eqn 3)

Table I lists q = 100, at which every weight underflows to ~1e-44 and no
activity gradient forms at all.  See PAPER_ERRATA_IGBNN.md item 1; the
constructor argument ``a`` carries the paper's q and defaults to 2.

Usage
-----
    from igbnnmu.open_igbnnmu import IGBNN_mu, make_grid

    grid    = make_grid(30, 30, obstacle_chance=0.05, seed=42)
    planner = IGBNN_mu(grid, n_robots=3)
    paths, stats = planner.run()

Tick-driven, for external renderers and ROS2 action servers::

    planner.reset()
    while planner.step():
        draw(planner.render_state())
    paths, stats = planner.final_result()

``paths`` is {robot_id: [(gx, gy), ...]} in global coordinates.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["IGBNN_mu", "make_grid", "REF_GRIDS"]

Cell = Tuple[int, int]

OBSTACLE = -1.0
UNVISITED = 1.0
VISITED = 0.0

# ============================================================================
#  UTILITIES
# ============================================================================


def make_grid(H: int, V: int, obstacle_chance: float = 0.05,
              seed: Optional[int] = None) -> np.ndarray:
    """Random H x V grid.  -1.0 = obstacle, 1.0 = free.

    Returns a 2-D array.  IGBNN_mu also accepts the legacy (H, V, 1) shape.
    """
    rng = np.random.default_rng(seed)
    return np.where(rng.random((H, V)) < obstacle_chance, OBSTACLE, UNVISITED)


def _ref_room(H: int, V: int, walls: Sequence[Tuple[int, int, int, int]]
              ) -> np.ndarray:
    g = np.full((H, V), UNVISITED)
    for (r0, r1, c0, c1) in walls:
        g[r0:r1, c0:c1] = OBSTACLE
    return g


#: Deterministic reference maps used by the smoke tests and benchmarks.
REF_GRIDS: Dict[int, np.ndarray] = {
    # 0 -- empty room
    0: np.full((20, 20), UNVISITED),
    # 1 -- two pillars
    1: _ref_room(20, 20, [(5, 9, 5, 9), (12, 16, 12, 16)]),
    # 2 -- corridor spine with a doorway
    2: _ref_room(24, 24, [(11, 13, 0, 10), (11, 13, 14, 24)]),
    # 3 -- comb / office partitions
    3: _ref_room(24, 24, [(4, 20, 6, 7), (4, 20, 12, 13), (4, 20, 18, 19)]),
}

# ============================================================================
#  IGBNN_mu
# ============================================================================


class IGBNN_mu:
    """Standalone IGBNN-mu complete coverage path planner.

    Parameters
    ----------
    full_grid : array_like, shape (H, V) or (H, V, 1)
        Environment map.  -1.0 = obstacle, 1.0 = free, 0.0 = pre-visited.
    n_robots : int
        Number of robots.
    E : float
        Large constant for the GBNN external bias term (Eqn 5).  Default 100.
    R : float
        Receptive field radius for the weight w_ij (Eqn 4).  Default 2.
    a : float
        Decay constant in the weight formula (Eqn 4).  Default 2.0.
        Corresponds to the symbol ``q`` in the paper.
    b : float
        Slope of the activity function on [0, 1) (Eqn 2).  Default 0.7.
        Corresponds to the symbol ``a`` in the paper; renamed here to avoid
        collision with the weight-decay parameter.
    mini_step_cap : int or None
        Hard cap on inner-loop ticks per minigraph.  Deterministic safety
        net that replaces the original wall-clock timeout.  None (default)
        scales it with the minigraph area: 8 * h * v + 200.
    decomposition : {"hcf", "lattice"}
        Minigraph sizing strategy -- see ``_pick_mini_size``.  "hcf" is the
        original behaviour and stays the default; "lattice" targets the
        n x n minigraph lattice of Algorithm 4 and avoids the 1x1 collapse
        that "hcf" produces whenever gcd(side, n) == 1.
    transit_step_cap : int
        Hard cap on ticks for one inter-minigraph A* transit.  Default 500.
    mop_up : bool
        After the serpentine sweep, revisit any free cell that is still
        unvisited but globally reachable.  This is what makes the coverage
        completeness claim (thesis Appendix A) hold when the minigraph
        decomposition splits a region whose only connection runs through a
        neighbouring minigraph.  Default True.
    visualize : bool
        Render the full grid every tick with matplotlib.  Default False.
        matplotlib is imported lazily, only when this is first used, so the
        module stays importable (and fast) on headless machines.
    viz_interval : int
        Render every N ticks when ``visualize`` is on.  Default 1.
    viz_pause : float
        Seconds handed to ``plt.pause`` after each live frame.  Default
        0.05, which keeps a few-hundred-step sweep watchable without
        dragging it out.
    mini_h, mini_v : int or None
        Override the minigraph height / width.  If None, auto-computed from
        Rule 1 + Rule 2 (see ``_pick_mini_size``).
    seed : int or None
        Seed for the tie-break jitter.  None keeps the planner fully
        deterministic (no jitter is used at all); an int makes ties broken
        by a seeded RNG, which is occasionally useful for benchmark spread.
    """

    COLORS = [
        "blue", "green", "red", "cyan", "magenta",
        "orange", "purple", "brown", "pink", "olive",
    ]
    _DIRS8 = [(0, -1), (0, 1), (-1, 0), (1, 0),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Diagonal move cost for the transit A*.  Unit cardinal cost keeps the
    # octile heuristic below admissible.
    _SQRT2 = math.sqrt(2.0)

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        full_grid,
        n_robots: int,
        E: float = 100.0,
        R: float = 2.0,
        a: float = 2.0,
        b: float = 0.7,
        mini_step_cap: Optional[int] = None,
        transit_step_cap: int = 500,
        decomposition: str = "paper",
        mop_up: bool = True,
        visualize: bool = False,
        viz_interval: int = 1,
        viz_pause: float = 0.05,
        mini_h: Optional[int] = None,
        mini_v: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:

        g = np.asarray(full_grid, dtype=float)
        if g.ndim == 3:
            if g.shape[2] != 1:
                raise ValueError(
                    f"full_grid must have a trailing dim of 1, got {g.shape}")
            g = g[:, :, 0]
        elif g.ndim != 2:
            raise ValueError(
                f"full_grid must be 2-D or 3-D, got shape {g.shape}")

        bad = ~np.isin(g, (OBSTACLE, VISITED, UNVISITED))
        if bad.any():
            raise ValueError(
                "full_grid must contain only -1.0, 0.0 or 1.0; "
                f"found {np.unique(g[bad])[:5]}")

        if n_robots < 1:
            raise ValueError(f"n_robots must be >= 1, got {n_robots}")

        self.H, self.V = g.shape
        self.n = int(n_robots)
        self.E = float(E)
        self.R = float(R)
        self.a = float(a)
        self.b = float(b)
        if decomposition not in ("paper", "hcf", "lattice"):
            raise ValueError(
                f"decomposition must be 'paper', 'hcf' or 'lattice', "
                f"got {decomposition!r}")
        self.decomposition = decomposition
        # None -> scale with the minigraph area (set in _decompose).  A
        # flat cap is wrong by construction: the same number is either
        # far too loose for a 6x6 minigraph or far too tight for the
        # 50x50 single minigraph an HCF collapse produces.
        self._mini_step_cap_arg = mini_step_cap
        self.mini_step_cap = int(mini_step_cap) if mini_step_cap else 0
        self.transit_step_cap = int(transit_step_cap)
        self.mop_up = bool(mop_up)
        self.visualize = bool(visualize)
        self.viz_interval = max(1, int(viz_interval))
        self.viz_pause = max(1e-3, float(viz_pause))
        self._mini_h_override = mini_h
        self._mini_v_override = mini_v
        self._rng = np.random.default_rng(seed) if seed is not None else None

        # Pristine input, kept for stats / reset().  Padding (Section IV-C)
        # may grow this in _plan_partition; orig_shape keeps what the caller
        # handed us so reports stay in the caller's terms.
        self.full_grid = g.copy()
        self.orig_shape: Tuple[int, int] = (self.H, self.V)

        # -------- the two arrays that must never be conflated ----------
        # Ground-truth coverage state.  Monotone 1.0 -> 0.0.
        self.state = g.copy()
        # Neural activity over the current minigraph slice.  Allocated per
        # minigraph in _begin_minigraph().
        self._act: np.ndarray = np.zeros((0, 0), dtype=float)

        # Merged path plans per robot: {own_id: [(gx, gy), ...]}
        self.path_plans: Dict[int, List[Cell]] = {i: [] for i in range(self.n)}

        # Robot state rows (see module docstring).
        self.robots: List[list] = []

        # Precomputed 8-neighbour weight kernel (Eqn 4).  w depends only on
        # the offset, so it is the same for every cell.
        self._w8 = [self._weight_for_offset(dx, dy) for dx, dy in self._DIRS8]

        # Diagnostics.  Declared before _plan_partition() because padding
        # and collapse both warn during construction.
        self._warnings: List[str] = []

        # Section IV-C: the partition is computed once, as soon as the map
        # dimensions are known, and (H, V, u) is constant thereafter.  Doing
        # it here rather than in reset() means padding is already applied
        # before any caller reads self.H / self.V / self.full_grid.
        self._minis = []
        self._plan_partition()

        # ---- tick-driven execution state (populated by reset()) --------
        self._started = False
        self._done = False
        self._stop_reason: Optional[str] = None
        self._phase = "idle"          # idle|cover|transit|mopup|done
        self._step = 0                # global tick counter
        self._mini_step = 0           # ticks inside the current minigraph
        self._transit_step = 0        # ticks inside the current transit
        self._t0 = 0.0

        self._minis: List[Tuple[int, int]] = []   # (row_off, col_off)
        self._mini_shape: Tuple[int, int] = (0, 0)
        self._order: List[Tuple[int, int]] = []
        self._nr = self._nc = 0
        self._visit_idx = 0
        self._ro = self._co = 0
        self._reachable: Optional[np.ndarray] = None   # bool mask, mini-local
        self._trace: List[List[Cell]] = []
        self._robot_global: List[Cell] = []
        self._transit_paths: List[List[Cell]] = []
        self._transit_ptrs: List[int] = []
        self._transit_blocked: List[int] = []
        self._transit_targets: List[Cell] = []
        self._mopup_queue: List[Cell] = []

        self._n_mini_capped = 0
        self._n_mini_stalled = 0
        self._n_transit_capped = 0
        self._n_unreachable_skipped = 0
        self._n_minis_skipped = 0
        self._idle: set = set()

    # ==================================================================
    #  SECTION 1 -- GBNN CORE  (Equations 2-5)
    # ==================================================================

    def _weight_for_offset(self, dx: int, dy: int) -> float:
        """Connection weight for a neighbour at offset (dx, dy).  Eqn 4.

        w_ij = exp(-q * d^2)  for 0 < d <= R,  else 0.
        """
        d = math.hypot(dx, dy)
        return math.exp(-self.a * d * d) if 0.0 < d <= self.R else 0.0

    def _f(self, x: float) -> float:
        """Activity function f(x).  Eqn 2."""
        if x < 0.0:
            return -1.0
        if x >= 1.0:
            return 1.0
        return self.b * x

    def _bias(self, gx: int, gy: int) -> float:
        """External input I_ij.  Eqn 5.  Derived from STATE, never activity."""
        s = self.state[gx, gy]
        if s == UNVISITED:
            return self.E
        if s == OBSTACLE:
            return -self.E
        return 0.0

    def _update_activity(self) -> None:
        """One full neural-activity pass over the current minigraph.

        Vectorised: the neighbour sum is eight shifted copies of
        max(act, 0), each scaled by its (constant) offset weight.  This is
        the same arithmetic as the scalar reference implementation but runs
        in O(cells) numpy work instead of O(cells * 8) Python work.

        Note the pass is synchronous (all cells see the previous tick's
        activity), unlike the in-place reference update.  Synchronous
        updates make a tick order-independent and therefore reproducible.
        """
        act = self._act
        h, v = act.shape
        pos = np.maximum(act, 0.0)
        s = np.zeros((h, v), dtype=float)

        for (dx, dy), w in zip(self._DIRS8, self._w8):
            if w == 0.0:
                continue
            # Shift `pos` so that s[i,j] accumulates pos[i+dx, j+dy].
            src_r = slice(max(0, dx), h + min(0, dx))
            dst_r = slice(max(0, -dx), h + min(0, -dx))
            src_c = slice(max(0, dy), v + min(0, dy))
            dst_c = slice(max(0, -dy), v + min(0, -dy))
            s[dst_r, dst_c] += w * pos[src_r, src_c]

        total = s + self._bias_field
        # f(): -1 below zero, b*x on [0,1), 1 at or above 1.
        out = np.where(total < 0.0, -1.0,
                       np.where(total >= 1.0, 1.0, self.b * total))
        self._act = out

    def _refresh_bias_field(self) -> None:
        """Recompute the I_ij field for the current minigraph from `state`."""
        sl = self.state[self._ro:self._ro + self._mini_shape[0],
                        self._co:self._co + self._mini_shape[1]]
        self._bias_field = np.where(sl == UNVISITED, self.E,
                                    np.where(sl == OBSTACLE, -self.E, 0.0))

    # ==================================================================
    #  SECTION 2 -- NEIGHBOUR GENERATION
    # ==================================================================

    def _in_mini(self, x: int, y: int) -> bool:
        h, v = self._mini_shape
        return 0 <= x < h and 0 <= y < v

    def _valid(self, x: int, y: int) -> bool:
        """True if local (x, y) is inside the minigraph and not an obstacle."""
        return (self._in_mini(x, y)
                and self.state[x + self._ro, y + self._co] != OBSTACLE)

    def _robot_locals(self) -> List[Cell]:
        return [(r[0], r[1]) for r in self.robots]

    def _nav_nbrs(self, robot: list) -> List[Cell]:
        """Morphology-aware neighbours for waypoint selection.

        Cardinal step size is scaled by the fusion count along that axis --
        this is the *neuron skipping* component: a formation fused n-deep
        vertically sweeps n rows at once, so its host advances n rows and
        the intervening rows are covered by the followers riding along.

        Cells currently occupied by another robot are excluded.
        """
        cx, cy = robot[0], robot[1]
        step_x = len(set(robot[7])) + 1     # vertical fuse count + 1
        step_y = len(set(robot[8])) + 1     # horizontal fuse count + 1
        occ = set(self._robot_locals())
        offsets = [(0, -step_y), (0, step_y), (-step_x, 0), (step_x, 0),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
        nbs = [(cx + dx, cy + dy) for dx, dy in offsets
               if self._valid(cx + dx, cy + dy)
               and (cx + dx, cy + dy) not in occ]
        if nbs:
            return nbs
        # Boxed in.  Hold position -- but only if the robot is actually
        # inside the minigraph, because the caller indexes the activity
        # field with whatever comes back.  A robot stranded outside (a
        # capped transit can leave one there) is parked on its nearest
        # in-bounds free cell instead of indexing out of range.
        if self._valid(cx, cy):
            return [(cx, cy)]
        return [self._nearest_free_local((cx, cy))]

    def _nearest_free_local(self, local: Cell) -> Cell:
        """Nearest non-obstacle cell inside the current minigraph (local)."""
        h, v = self._mini_shape
        cx = min(max(local[0], 0), h - 1)
        cy = min(max(local[1], 0), v - 1)
        if self.state[cx + self._ro, cy + self._co] != OBSTACLE:
            return (cx, cy)
        best, best_d = None, math.inf
        for i in range(h):
            for j in range(v):
                if self.state[i + self._ro, j + self._co] == OBSTACLE:
                    continue
                d = (i - cx) ** 2 + (j - cy) ** 2
                if d < best_d:
                    best, best_d = (i, j), d
        if best is None:
            raise ValueError(
                f"minigraph at offset ({self._ro}, {self._co}) is entirely "
                f"obstacle -- no free cell to place a robot on")
        return best

    def _free_local_excluding(self, exclude: set) -> Optional[Cell]:
        """A free local cell in the current minigraph, or None if it is full.

        Returns None rather than raising: running out of seats is the normal
        ``i_unoccupied < n`` case of Section IV-C, not an error.
        """
        h, v = self._mini_shape
        for i in range(h):
            for j in range(v):
                if ((i, j) not in exclude
                        and self.state[i + self._ro, j + self._co] != OBSTACLE):
                    return (i, j)
        return None

    # ==================================================================
    #  SECTION 3 -- XFM FUNCTION  (Algorithm 3)
    # ==================================================================

    def _xfm(self, idx: int) -> None:
        """Detect adjacent robots and update fusion state (S_h, S_v).

        Vertically adjacent  -> record in S_v (x_fused), neighbour follows.
        Horizontally adjacent-> record in S_h (y_fused), neighbour follows.
        Diagonally adjacent  -> de-fuse (a diagonal pair is not a valid
                                rigid morphology).

        Algorithm 3 line 4 guards on ``i_c < i``: a robot only ever inspects
        neighbours whose index is HIGHER than its own.  That asymmetry is
        what makes the lower index the host -- Section III-C, "the attached
        robot with a greater index number will follow the host robot's
        direction... the combined robot identifies its index with the robot
        with the smallest index".  Dropping the guard would let two adjacent
        robots each claim the other, and the pair would deadlock.

        Fused lists hold *own_ids*, and every lookup below uses own_id, not
        the list index.  The two coincide today because robots are created
        in own_id order, but relying on that made the original de-fuse
        branch silently wrong the moment a robot was reordered or removed.

        Note on Algorithm 3 lines 8 and 12 as published: they read
        ``Update S_alpha,ic <- S_alpha,i``, which assigns the LOWER-indexed
        robot's alpha from the higher-indexed one -- the host would follow
        its own follower.  Section III-C's prose states the opposite, and
        this implementation follows the prose.  See PAPER_ERRATA_IGBNN.md.
        """
        r = self.robots[idx]
        my_id = r[6]
        cx, cy = r[0], r[1]

        for dx, dy in self._DIRS8:
            nx, ny = cx + dx, cy + dy
            if not self._valid(nx, ny):
                continue
            for other in self.robots:
                oid = other[6]
                # Algorithm 3 line 4: `i_c < i` -- higher indices only.
                if oid <= my_id or (nx, ny) != (other[0], other[1]):
                    continue
                if abs(dx) == 1 and dy == 0:
                    # Vertically adjacent -- S_v
                    if oid not in r[7]:
                        r[7].append(oid)
                    if oid in r[8]:
                        r[8].remove(oid)
                    if other[5] == oid:          # only claim a free robot
                        other[5] = my_id
                elif dx == 0 and abs(dy) == 1:
                    # Horizontally adjacent -- S_h
                    if oid not in r[8]:
                        r[8].append(oid)
                    if oid in r[7]:
                        r[7].remove(oid)
                    if other[5] == oid:
                        other[5] = my_id
                else:
                    # Diagonal -- not a valid fused morphology, de-fuse.
                    if oid in r[7]:
                        r[7].remove(oid)
                    if oid in r[8]:
                        r[8].remove(oid)
                    if other[5] == my_id:
                        other[5] = oid

    def _robot_by_id(self, own_id: int) -> Optional[list]:
        for r in self.robots:
            if r[6] == own_id:
                return r
        return None

    # ==================================================================
    #  SECTION 4 -- WAYPOINT SELECTION  (Eqn 6)
    # ==================================================================

    def _next_wp(self, idx: int, nbs: Sequence[Cell]) -> Cell:
        """Pick the next waypoint: highest activity, with tie-breaks.

        Eqn 6 selects argmax of the neighbour activity.  Ties are common --
        every unvisited cell saturates f() at exactly 1.0 -- and the
        original implementation resolved them with `list.index(max(...))`,
        which always returns the first candidate and so biased every robot
        toward -y.  That produces ragged sweeps and needless revisits.

        Tie-break order, applied only among the joint-argmax candidates:
          1. prefer a genuinely UNVISITED cell over a visited one
             (a visited cell can also reach activity 1.0 when it is
             surrounded by enough unvisited neighbours);
          2. prefer continuing in the current heading (momentum), which
             produces long straight boustrophedon runs;
          3. seeded RNG if `seed` was given, else first in offset order
             so the planner stays fully deterministic by default.

        No step-cost term here, deliberately.  GBNN selects
        ``max(x_j + c * y_j)`` -- the sibling implementation in this repo
        (``common/replicated_gbnn.py``) uses ``y_j = 1 - d/sqrt(2)`` at
        ``c = 0.01``, which pays a cardinal neighbour and not a diagonal
        one, and is why GBNN sweeps in pure axis-aligned boustrophedon
        rows.  Eqn 6 is a bare ``max(n_k, ..., n_lambda-1)`` and Table I
        lists only E, q, a, R and lambda, so there is no such constant to
        inherit and none is invented here.

        The visible consequence, and why it is not a defect: on the same
        6x6 m area GBNN with one robot makes 0% diagonal moves at every
        map size tried, while IGBNN-mu with five makes 29% (12-18% on
        larger maps), so its trail reads as oblique strokes rather than
        rows.  Adding GBNN's term does NOT explain the gap -- measured at
        ``c = 0.01`` it changes nothing at all, and even at ``c = 3.0``
        only 29% -> 24%.  The diagonals are mostly FORCED: instrumenting
        every diagonal step to ask whether an unvisited, unoccupied
        cardinal neighbour existed at that moment gives 27 of 36 forced
        on the 8x8 case -- the cardinal neighbours were already covered
        or standing on a team-mate.  Rule 2 forces ``u >= n``, so five
        robots on this area share a single 5x5 minigraph and box each
        other in constantly; a lone GBNN robot in open space always has
        unvisited ground straight ahead.  Overriding the argmax to
        suppress the remaining chosen diagonals would be a real
        deviation from Eqn 6, so it is not done.
        """
        r = self.robots[idx]
        acts = [self._act[x, y] for (x, y) in nbs]
        best = max(acts)
        cand = [i for i, v in enumerate(acts) if v >= best - 1e-12]

        if len(cand) > 1:
            unvis = [i for i in cand
                     if self.state[nbs[i][0] + self._ro,
                                   nbs[i][1] + self._co] == UNVISITED]
            if unvis:
                cand = unvis

        if len(cand) > 1 and (r[3], r[4]) != (0, 0):
            # Momentum: same sign of travel on both axes.
            def aligned(i: int) -> bool:
                dx = nbs[i][0] - r[0]
                dy = nbs[i][1] - r[1]
                return (np.sign(dx) == np.sign(r[3])
                        and np.sign(dy) == np.sign(r[4]))
            keep = [i for i in cand if aligned(i)]
            if keep:
                cand = keep

        if len(cand) > 1 and self._rng is not None:
            pick = cand[int(self._rng.integers(len(cand)))]
        else:
            pick = cand[0]
        return nbs[pick]

    def _commit_move(self, idx: int, cell: Cell) -> None:
        """Move robot `idx` to local `cell`, mark it visited, log the step."""
        r = self.robots[idx]
        r[3] = cell[0] - r[0]
        r[4] = cell[1] - r[1]
        r[0], r[1] = cell
        gx, gy = cell[0] + self._ro, cell[1] + self._co
        if self.state[gx, gy] != OBSTACLE:
            self.state[gx, gy] = VISITED
        self.path_plans[r[6]].append((gx, gy))

    def _detach(self, idx: int) -> None:
        """Detach robot `idx` from its host and clear its own fusion state."""
        r = self.robots[idx]
        my_id = r[6]
        host = self._robot_by_id(r[5])
        if host is not None and host[6] != my_id:
            for slot in (7, 8):
                if my_id in host[slot]:
                    host[slot].remove(my_id)
        r[5] = my_id
        r[7] = []
        r[8] = []

    # ==================================================================
    #  SECTION 5 -- MINIGRAPH DECOMPOSITION  f(Grid, u = n)
    # ==================================================================

    @staticmethod
    def _valid_divisors(length: int, min_size: int) -> List[int]:
        """Divisors of `length` that are >= min_size, ascending."""
        return sorted(d for d in range(1, length + 1)
                      if length % d == 0 and d >= min_size)

    def _plan_partition(self) -> None:
        """Choose the partition once, then hold it fixed.  Section IV-C.

        The paper is explicit that this happens exactly once:

            "Both map discretization and minigraph sizing are fixed for a
            given map and robot population.  No default size is assumed a
            priori; the minigraph size is initialized as None and computed
            once map dimensions are available, after which (H, V, u)
            remains constant for the remainder of execution."

        and that the partition is square:

            "To maintain square partitions, we enforce H = V and define the
            number of minigraphs as n_mu = sqrt(u) with u = max(H, V) and
            u >= n to ensure sufficient initialization capacity."

        Reading of ``n_mu = sqrt(u)`` (confirmed with the author):
        ``u`` is the minigraph edge length and equals ``sqrt(L)`` for a
        padded side ``L``, so the map divides into a ``u x u`` lattice of
        ``u x u`` tiles and the per-axis minigraph count is ``sqrt(L) = u``.
        This is the arrangement drawn in Fig. 6, and it makes Rule 2 read
        directly: the tile edge ``u`` must seat all ``n`` robots.

        The side is therefore padded up to the next perfect square (30 ->
        36 with u=6; 50 -> 64 with u=8), and further if ``u < n`` would
        otherwise violate Rule 2:

            "Candidate tile sizes (H, V) are selected such that (i) the
            grid can be partitioned with minimal padding and (ii) each
            minigraph provides sufficient boundary capacity for robot
            initialization (Rule 2).  If the grid is not evenly divisible,
            the shorter edge is padded... padded cells are treated as
            unknown and conservatively marked as occupied."

        Padding is applied to ``self.full_grid`` here, so every downstream
        index is already in padded coordinates.  ``self.orig_shape`` keeps
        the caller's dimensions for reporting.
        """
        # An explicit mini_h / mini_v always wins, whatever the strategy.
        # Silently ignoring a caller's override would be the worst kind of
        # surprise: the run succeeds, on a partition they did not ask for.
        if self._mini_h_override is not None or self._mini_v_override is not None:
            if self.decomposition == "paper":
                self._warn(
                    f"mini_h/mini_v override supplied, so the Section IV-C "
                    f"square-partition rule is bypassed for this run")
            h, method_h, v, method_v = self._pick_legacy_sizes()
        elif self.decomposition == "paper":
            u, pad_r, pad_c, method = self._pick_u_paper()
            if pad_r or pad_c:
                self._pad_grid(pad_r, pad_c)
                self._warn(
                    f"grid padded from {self.orig_shape[0]}x"
                    f"{self.orig_shape[1]} to {self.H}x{self.V} (= {u}^2) so "
                    f"a {u}x{u} lattice of square {u}x{u} minigraphs tiles "
                    f"it exactly; the {pad_r} added row(s) and {pad_c} added "
                    f"column(s) are marked occupied (Section IV-C)")
            h = v = u
            method_h = method_v = method
        else:
            h, method_h, v, method_v = self._pick_legacy_sizes()

        self._mini_shape = (h, v)
        self._nr = self.H // h
        self._nc = self.V // v
        self._minis = [(mr * h, mc * v)
                       for mr in range(self._nr)
                       for mc in range(self._nc)]
        if not self._mini_step_cap_arg:
            self.mini_step_cap = 24 * h * v + 500

        if self._nr == 1 and self._nc == 1 and self.n > 1:
            self._warn(
                f"decomposition collapsed to a single {h}x{v} minigraph "
                f"(strategy {self.decomposition!r}, n={self.n}): the "
                f"minigraph mechanism is inactive for this map/robot-count "
                f"pair.  decomposition='paper' avoids this.")

        self._decomp_note = (
            f"{self._nr}x{self._nc} minigraphs ({len(self._minis)} total), "
            f"each {h}x{v}  [h:{method_h}  v:{method_v}  "
            f"Rule1 ok  Rule2 ok  n={self.n}<=min({h},{v})]"
        )

    def _pick_u_paper(self) -> Tuple[int, int, int, str]:
        """Return (u, pad_rows, pad_cols, label) for the paper's rule.

        Both axes are padded to the same square side ``L = u*u``, where

            u = smallest integer with  u*u >= max(i, j)  and  u >= n

        The first condition is "minimal padding" -- u is the smallest tile
        edge whose square still contains the map, so no smaller choice
        avoids more padding.  The second is Rule 2: the tile edge has to
        seat every robot along an entry edge, so a large team forces a
        coarser lattice (and correspondingly more padding).
        """
        i, j = self.H, self.V
        side = max(i, j)

        # Tile edge: the square-lattice ideal is u = ceil(sqrt(side)), which
        # makes the lattice u x u and n_mu = sqrt(L) exactly whenever `side`
        # is already a perfect square.  Rule 2 raises it further when the
        # team needs a longer entry edge.
        u = max(math.isqrt(side - 1) + 1 if side > 1 else 1, 1)
        u = max(u, self.n)

        # Padded side: the smallest multiple of u that still contains the
        # map, NOT u*u.  Criterion (i) -- "minimal padding" -- is stated
        # first in Section IV-C and has to win here: forcing L = u*u pads a
        # 6x6 region out to 25x25 when five robots make u = 5, so 94% of the
        # grid is padding and 21 of 25 minigraphs are fully blocked.
        # k = ceil(side / u) keeps the tiles square and the lattice square
        # while adding the least padding that tiles the map.
        k = -(-side // u)
        L = k * u
        return (u, L - i, L - j,
                f"paper(u={u}, {k}x{k} lattice of {u}x{u}, L={L})")

    def _pad_grid(self, pad_r: int, pad_c: int) -> None:
        """Grow full_grid by pad_r rows / pad_c cols of OBSTACLE cells.

        Section IV-C: padded cells "are treated as unknown and
        conservatively marked as occupied".  Marking them OBSTACLE rather
        than VISITED keeps them out of both the free-cell denominator and
        the reachable set, so padding never inflates the coverage figure.
        """
        H, V = self.full_grid.shape
        g = np.full((H + pad_r, V + pad_c), OBSTACLE, dtype=float)
        g[:H, :V] = self.full_grid
        self.full_grid = g
        self.state = g.copy()
        self.H, self.V = g.shape

    def _pick_legacy_sizes(self) -> Tuple[int, str, int, str]:
        """Non-paper strategies, kept for reproducing earlier runs."""
        n = self.n
        if self._mini_h_override is not None:
            h = int(self._mini_h_override)
            if h <= 0 or self.H % h != 0:
                raise ValueError(
                    f"Rule 1 violated: H={self.H} not divisible by mini_h={h}.")
            if n > h:
                raise ValueError(f"Rule 2 violated: n={n} > mini_h={h}.")
            method_h = "override"
        else:
            h, method_h = self._pick_mini_size(self.H, n)

        if self._mini_v_override is not None:
            v = int(self._mini_v_override)
            if v <= 0 or self.V % v != 0:
                raise ValueError(
                    f"Rule 1 violated: V={self.V} not divisible by mini_v={v}.")
            if n > v:
                raise ValueError(f"Rule 2 violated: n={n} > mini_v={v}.")
            method_v = "override"
        else:
            v, method_v = self._pick_mini_size(self.V, n)
        return h, method_h, v, method_v

    def _pick_mini_size(self, length: int, n: int) -> Tuple[int, str]:
        """Choose the minigraph edge length for one axis.

        Two strategies, selected by the ``decomposition`` constructor
        argument.

        ``"hcf"`` (default, preserves the original behaviour)
            h = length // gcd(length, n).  Larger minigraphs mean more IGBNN
            coverage per minigraph and fewer A* transits.

            Caveat, and the reason the ``"lattice"`` option exists: whenever
            gcd(length, n) == 1 this returns h = length, i.e. a 1x1 lattice
            -- one minigraph covering the whole map, with the minigraph
            mechanism effectively switched off.  That is not a rare corner:
            it fires for (H=40, n=3), (H=50, n=3), (H=36, n=5) and for every
            map whose side is not a multiple of 7 at n=7.  ``_decompose``
            records a warning when it happens so the collapse is visible in
            ``stats["warnings"]`` rather than silent.

        ``"lattice"``
            Aim for the n x n lattice that Algorithm 4 describes ("decompose
            into n^2 minigraphs"): pick the divisor of `length` closest to
            length / n.  Falls back the same way when Rule 2 bites.

        Rule 2 fallback (both strategies): if n > h the robots do not fit
        along the entry edge, so use the smallest divisor of `length` that
        is >= n.
        """
        if self.decomposition == "lattice":
            cands = self._valid_divisors(length, n)
            if cands:
                target = length / n
                best = min(cands, key=lambda d: (abs(d - target), d))
                return best, "lattice"
        else:
            h_hcf = length // math.gcd(length, n)
            if n <= h_hcf:
                return h_hcf, "HCF"

        cands = self._valid_divisors(length, n)
        if not cands:
            raise ValueError(
                f"Rule 2: no valid minigraph size exists for length={length}, "
                f"n={n}.  The grid dimension needs a divisor >= n "
                f"(divisors of {length}: "
                f"{sorted(d for d in range(1, length + 1) if length % d == 0)})."
            )
        return cands[0], "fallback(smallest>=n)"

    def _decompose(self) -> None:
        """Algorithm 4 line 3, ``Compute Minigraph <- f(Grid, u)``.

        The partition itself was fixed by ``_plan_partition()`` at
        construction, per Section IV-C ("(H, V, u) remains constant for the
        remainder of execution").  This is the re-entry point kept for the
        callers that decompose eagerly to inspect the lattice.
        """
        if not self._minis:
            self._plan_partition()

    def _mini_is_blocked(self, ro: int, co: int) -> bool:
        """True when a minigraph has no free cell at all.

        Section IV-C: "Fully blocked minigraphs are skipped... while
        partially free ones are still planned."  A tile that is entirely
        obstacle (common once padding is in play, since padded strips are
        marked occupied) has nothing to cover and no cell to stand on, so
        entering it at all is wasted work.
        """
        h, v = self._mini_shape
        return not bool(np.any(self.state[ro:ro + h, co:co + v] != OBSTACLE))

    def _serpentine(self) -> List[Tuple[int, int]]:
        """Boustrophedon visit order over the nr x nc minigraph lattice."""
        order = []
        skipped = 0
        for r in range(self._nr):
            cols = range(self._nc) if r % 2 == 0 else range(self._nc - 1, -1, -1)
            for c in cols:
                ro, co = self._minis[r * self._nc + c]
                if self._mini_is_blocked(ro, co):
                    skipped += 1
                    continue
                order.append((r, c))
        if skipped:
            self._n_minis_skipped = skipped
            self._warn(
                f"{skipped} fully blocked minigraph(s) skipped "
                f"(Section IV-C)")
        if not order:
            raise ValueError(
                "every minigraph is fully blocked -- the map has no free "
                "cell to cover")
        return order

    @staticmethod
    def _entry_dir(prev_rc: Tuple[int, int],
                   curr_rc: Tuple[int, int]) -> str:
        """Entry edge of curr_rc given the direction of travel from prev_rc."""
        dr = curr_rc[0] - prev_rc[0]
        dc = curr_rc[1] - prev_rc[1]
        if dr == 0 and dc > 0:
            return "left"
        if dr == 0 and dc < 0:
            return "right"
        if dr > 0:
            return "top"
        return "bottom"

    def _edge_cells(self, ro: int, co: int, edge: str) -> List[Cell]:
        """Global coords of every non-obstacle cell on `edge` of a minigraph."""
        h, v = self._mini_shape
        if edge == "left":
            cells = [(i + ro, co) for i in range(h)]
        elif edge == "right":
            cells = [(i + ro, co + v - 1) for i in range(h)]
        elif edge == "top":
            cells = [(ro, j + co) for j in range(v)]
        else:  # bottom
            cells = [(ro + h - 1, j + co) for j in range(v)]
        return [c for c in cells if self.state[c] != OBSTACLE]

    def _even_spacing(self, cells: Sequence[Cell], n: int,
                      fallback: Optional[Sequence[Cell]] = None) -> List[Cell]:
        """Pick n DISTINCT positions spread evenly along `cells`.

        The original formula ``cells[round(k*(L-1)/(n-1))]`` collapses onto
        the same cell whenever L < n (e.g. L=2, n=3 -> indices 0, 0, 1 under
        banker's rounding), which puts two robots on one cell and trips the
        collision assertion.  Here the evenly spaced indices are taken
        first, then any duplicate is nudged to the nearest unused index; if
        the edge genuinely has fewer than n free cells the shortfall is
        drawn from `fallback`.

        `fallback` MUST be confined to the target minigraph.  An earlier
        version padded from the 8-neighbourhood of the edge cells, which
        reaches one row/column *outside* the minigraph -- robots were then
        sent to a cell beyond the box they were supposed to be entering,
        arrived successfully, and were flagged as stranded on entry.
        """
        if n <= 0:
            return []
        pool = list(cells)
        # Section IV-C: "If a minigraph contains fewer unoccupied cells than
        # robots (i_unoccupied < n), only i_unoccupied robots are assigned
        # and the remainder idle until the next minigraph."  Return the
        # short list; the caller idles whoever is left over.
        avail = len(set(pool) | set(fallback or ()))
        if avail < n:
            return list(dict.fromkeys(list(pool) + list(fallback or ())))
        if not pool:
            raise ValueError(
                "No free cells on the entry edge -- the minigraph is fully "
                "blocked along that edge.  Callers should use "
                "_entry_targets(), which falls back to the minigraph "
                "interior instead of failing.")

        L = len(pool)
        if n == 1:
            return [pool[L // 2]]

        raw = [int(round(k * (L - 1) / (n - 1))) for k in range(n)]
        used: set = set()
        out: List[Cell] = []
        for idx in raw:
            if idx not in used:
                used.add(idx)
                out.append(pool[idx])
                continue
            # Nudge outward to the nearest unused index.
            placed = False
            for delta in range(1, L):
                for cand in (idx + delta, idx - delta):
                    if 0 <= cand < L and cand not in used:
                        used.add(cand)
                        out.append(pool[cand])
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                out.append(None)   # type: ignore[arg-type]

        # Fill any shortfall from the (minigraph-confined) fallback pool.
        if any(c is None for c in out):
            taken = {c for c in out if c is not None}
            it = iter([c for c in (fallback or ()) if c not in taken])
            filled: List[Cell] = []
            for c in out:
                if c is not None:
                    filled.append(c)
                    continue
                nxt = next(it, None)
                while nxt is not None and nxt in taken:
                    nxt = next(it, None)
                if nxt is None:
                    break            # short list; caller idles the remainder
                taken.add(nxt)
                filled.append(nxt)
            out = filled

        return out

    def _entry_targets(self, ro: int, co: int, edge: str) -> List[Cell]:
        """Entry positions for the minigraph at (ro, co), entered via `edge`.

        Normally these are n evenly spaced free cells along that edge.  A
        wall can seal an entire edge, though (the corridor reference map
        does exactly this), and the original implementation raised in that
        case and aborted the whole run.  Here the fallback walks inward:
        the opposite edge, then the free interior cells nearest the edge.
        """
        h, v = self._mini_shape
        anchor = {"left": (ro + h // 2, co),
                  "right": (ro + h // 2, co + v - 1),
                  "top": (ro, co + v // 2),
                  "bottom": (ro + h - 1, co + v // 2)}[edge]
        interior = [(gx, gy)
                    for gx in range(ro, ro + h)
                    for gy in range(co, co + v)
                    if self.state[gx, gy] != OBSTACLE]
        if not interior:
            raise ValueError(
                f"minigraph at ({ro}, {co}) is entirely obstacle -- "
                f"no entry position exists")
        # Padding pool, nearest the intended edge first, and strictly inside
        # this minigraph.
        interior.sort(key=lambda c: (c[0] - anchor[0]) ** 2
                      + (c[1] - anchor[1]) ** 2)

        cells = self._edge_cells(ro, co, edge)
        if cells:
            return self._even_spacing(cells, self.n, fallback=interior)

        opposite = {"left": "right", "right": "left",
                    "top": "bottom", "bottom": "top"}[edge]
        cells = self._edge_cells(ro, co, opposite)
        if cells:
            self._warn(
                f"entry edge '{edge}' of minigraph at ({ro}, {co}) is fully "
                f"blocked; entering from '{opposite}' instead")
            return self._even_spacing(cells, self.n, fallback=interior)

        self._warn(
            f"both '{edge}' and '{opposite}' edges of the minigraph at "
            f"({ro}, {co}) are blocked; entering through the interior")
        return self._even_spacing(interior, self.n, fallback=interior)

    # ==================================================================
    #  SECTION 6 -- REACHABILITY
    # ==================================================================

    def _park_spare(self, k: int, exclude: set) -> List[Cell]:
        """k distinct free cells for robots that no minigraph can seat yet."""
        out: List[Cell] = []
        taken = set(exclude)
        for gx in range(self.H):
            for gy in range(self.V):
                if len(out) >= k:
                    return out
                if (gx, gy) in taken or self.state[gx, gy] == OBSTACLE:
                    continue
                taken.add((gx, gy))
                out.append((gx, gy))
        if len(out) < k:
            raise ValueError(
                f"the map has fewer free cells than robots (n={self.n})")
        return out

    def _flood(self, seeds: Sequence[Cell], bounds: Optional[
            Tuple[int, int, int, int]] = None) -> np.ndarray:
        """8-connected flood fill over non-obstacle cells (GLOBAL coords).

        Parameters
        ----------
        seeds : list of global (gx, gy)
        bounds : (r0, r1, c0, c1) half-open box to confine the fill, or None
                 for the whole map.

        Returns a bool mask over the full map.
        """
        r0, r1, c0, c1 = bounds if bounds else (0, self.H, 0, self.V)
        mask = np.zeros((self.H, self.V), dtype=bool)
        stack = []
        for (gx, gy) in seeds:
            if (r0 <= gx < r1 and c0 <= gy < c1
                    and self.state[gx, gy] != OBSTACLE and not mask[gx, gy]):
                mask[gx, gy] = True
                stack.append((gx, gy))
        while stack:
            gx, gy = stack.pop()
            for dx, dy in self._DIRS8:
                nx, ny = gx + dx, gy + dy
                if (r0 <= nx < r1 and c0 <= ny < c1
                        and not mask[nx, ny]
                        and self.state[nx, ny] != OBSTACLE):
                    mask[nx, ny] = True
                    stack.append((nx, ny))
        return mask

    def _mini_incomplete(self) -> bool:
        """True while a REACHABLE unvisited cell remains in this minigraph.

        Restricting the completeness test to the reachable set is what makes
        the inner loop terminate on maps with enclosed pockets.  Without it
        a cell walled off inside the minigraph keeps `complete` false
        forever and the loop runs until the step cap, burning thousands of
        ticks for nothing.  Pockets skipped here are picked up later by the
        global mop-up pass, which can approach them from another minigraph.
        """
        h, v = self._mini_shape
        sl = self.state[self._ro:self._ro + h, self._co:self._co + v]
        if self._reachable is None:
            return bool(np.any(sl == UNVISITED))
        return bool(np.any((sl == UNVISITED) & self._reachable))

    # ==================================================================
    #  SECTION 7 -- A*  (global coordinates, octile heuristic)
    # ==================================================================

    def _astar(self, start: Cell, goal: Cell,
               blocked: Optional[set] = None) -> Optional[List[Cell]]:
        """8-connected A* over free cells.  Global coordinates.

        Diagonal steps cost sqrt(2) and the heuristic is the octile
        distance, so the heuristic is admissible and consistent.  The
        original used unit cost for diagonals with a Manhattan heuristic --
        inadmissible on an 8-connected grid, which let A* return paths that
        were not shortest and, worse, expand nodes in a misleading order.

        `blocked` is an optional set of extra impassable cells (used to
        route a transit around robots that have already parked on their
        targets and will not move again).

        Returns the path from start through goal inclusive, or None if the
        goal is unreachable.  It never fabricates a one-element path: a
        silent ``[goal]`` return is a teleport through walls, and the caller
        cannot tell it apart from a legitimate zero-length move.
        """
        if start == goal:
            return [start]
        if self.state[goal] == OBSTACLE or self.state[start] == OBSTACLE:
            return None
        block = blocked or ()

        def h(p: Cell) -> float:
            dx = abs(p[0] - goal[0])
            dy = abs(p[1] - goal[1])
            return (dx + dy) + (self._SQRT2 - 2.0) * min(dx, dy)

        heap: List[Tuple[float, int, Cell]] = [(h(start), 0, start)]
        came: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}
        closed: set = set()
        counter = 1

        while heap:
            _, _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return list(reversed(path))
            for dx, dy in self._DIRS8:
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < self.H and 0 <= nb[1] < self.V):
                    continue
                if self.state[nb] == OBSTACLE or nb in closed or nb in block:
                    continue
                step = self._SQRT2 if (dx and dy) else 1.0
                ng = g_score[cur] + step
                if ng < g_score.get(nb, math.inf) - 1e-12:
                    came[nb] = cur
                    g_score[nb] = ng
                    heapq.heappush(heap, (ng + h(nb), counter, nb))
                    counter += 1
        return None

    def _nearest_reachable(self, start: Cell,
                           targets: Sequence[Cell]) -> Optional[Cell]:
        """First target in `targets` that A* can actually reach from start."""
        for t in targets:
            if self._astar(start, t) is not None:
                return t
        return None

    # ==================================================================
    #  SECTION 8 -- TICK-DRIVEN API  (reset / step / final_result)
    # ==================================================================
    #
    #  Mirrors the GBNN_H contract so external drivers -- pygame Mode 3,
    #  ROS2 action servers, headless benchmarks -- advance the planner on
    #  their own clock instead of blocking inside run().
    #
    #      planner = IGBNN_mu(grid, n_robots=3)
    #      planner.reset()
    #      while planner.step():
    #          render(planner.render_state())
    #      paths, stats = planner.final_result()

    def reset(self) -> None:
        """Prepare the planner for tick-driven execution.

        Restores `state` from the pristine input, decomposes the map, seeds
        the robots on the entry edge of the first minigraph and runs the
        first activity pass.  `step()` is legal only after this call.
        """
        self._t0 = time.time()
        self.state = self.full_grid.copy()
        self.path_plans = {i: [] for i in range(self.n)}
        self._step = 0
        self._mini_step = 0
        self._transit_step = 0
        self._visit_idx = 0
        # Keep construction-time warnings (padding, collapse) -- they
        # describe the partition, which reset() does not recompute.
        self._warnings = [w for w in self._warnings
                          if "padded" in w or "collapsed" in w
                          or "skipped" in w]
        self._n_mini_capped = 0
        self._n_mini_stalled = 0
        self._n_transit_capped = 0
        self._n_unreachable_skipped = 0
        self._n_idle_assignments = 0
        self._idle = set()
        self._done = False
        self._stop_reason = None
        self._mopup_queue = []

        self._decompose()
        self._order = self._serpentine()

        # Seed robots on the TOP edge of the first non-blocked minigraph.
        ro, co = self._minis[self._order[0][0] * self._nc + self._order[0][1]]
        starts = self._entry_targets(ro, co, "top")

        # Section IV-C: a minigraph with fewer free cells than robots seats
        # only as many as it can.  The remainder are parked on free cells
        # elsewhere on the map and idle until a minigraph has room.
        if len(starts) < self.n:
            self._warn(
                f"first minigraph seats only {len(starts)} of {self.n} "
                f"robots; the rest start idle (Section IV-C)")
            starts = starts + self._park_spare(self.n - len(starts),
                                               exclude=set(starts))

        self.robots = [[gx, gy, 0, 0, 0, i, i, [], []]
                       for i, (gx, gy) in enumerate(starts)]
        self._robot_global = list(starts)
        for i, (gx, gy) in enumerate(starts):
            if self.state[gx, gy] != OBSTACLE:
                self.state[gx, gy] = VISITED
            self.path_plans[i].append((gx, gy))

        self._started = True
        self._begin_minigraph(0)

        if self.visualize:
            self._viz("init")

    def _begin_minigraph(self, visit_idx: int) -> None:
        """Enter minigraph `visit_idx`: convert to local coords, seed fields."""
        self._visit_idx = visit_idx
        rc = self._order[visit_idx]
        self._ro, self._co = self._minis[rc[0] * self._nc + rc[1]]
        h, v = self._mini_shape

        # Global -> local, and reset every robot to autonomous.  Algorithm 4
        # re-forms morphologies from scratch inside each minigraph.
        for r in self.robots:
            r[0] -= self._ro
            r[1] -= self._co
            r[5] = r[6]
            r[7] = []
            r[8] = []
            r[3] = r[4] = 0

        # Partition the team into the robots this minigraph can seat and
        # the ones that idle through it.
        #
        # Section IV-C: "If a minigraph contains fewer unoccupied cells than
        # robots (i_unoccupied < n), only i_unoccupied robots are assigned
        # and the remainder idle until the next minigraph."
        #
        # A robot already standing on a free cell inside the tile is active.
        # One that is outside (a capped transit can leave it there, and
        # padding makes edge tiles mostly occupied) is moved in when there
        # is an unclaimed free cell for it, and idles otherwise -- rather
        # than being force-parked on top of the coverage, which is what the
        # pre-paper version did.
        self._idle = set()
        taken: set = set()
        for i, r in enumerate(self.robots):
            if self._valid(r[0], r[1]) and (r[0], r[1]) not in taken:
                taken.add((r[0], r[1]))
                continue
            cell = self._free_local_excluding(taken)
            if cell is None:
                self._idle.add(i)
                continue
            r[0], r[1] = cell
            taken.add(cell)

        if self._idle:
            self._n_idle_assignments += len(self._idle)
            self._warn(
                f"minigraph {self._order[visit_idx]} seats "
                f"{self.n - len(self._idle)} of {self.n} robots; "
                f"{len(self._idle)} idle through it (Section IV-C)")

        # Mark the active robots' entry cells visited.
        for i, r in enumerate(self.robots):
            if i in self._idle:
                continue
            gx, gy = r[0] + self._ro, r[1] + self._co
            if self.state[gx, gy] != OBSTACLE:
                self.state[gx, gy] = VISITED

        # Reachable set for this minigraph, seeded from the ACTIVE robots
        # only -- an idle robot sitting outside would otherwise seed a flood
        # that never touches this tile.
        seeds = [(r[0] + self._ro, r[1] + self._co)
                 for i, r in enumerate(self.robots) if i not in self._idle]
        full_mask = self._flood(
            seeds, bounds=(self._ro, self._ro + h, self._co, self._co + v))
        self._reachable = full_mask[self._ro:self._ro + h,
                                    self._co:self._co + v]

        sl = self.state[self._ro:self._ro + h, self._co:self._co + v]
        skipped = int(np.sum((sl == UNVISITED) & ~self._reachable))
        if skipped:
            self._n_unreachable_skipped += skipped

        self._act = np.where(sl == OBSTACLE, -1.0,
                             np.where(sl == UNVISITED, 1.0, 0.0))
        self._refresh_bias_field()
        self._update_activity()

        self._trace = [[(r[0], r[1])] * 10 for r in self.robots]
        self._mini_step = 0
        self._stale_ticks = 0
        self._last_remaining = int(np.sum(sl == UNVISITED))
        self._phase = "cover"

    # ---- one inner-loop tick (Algorithm 2) ---------------------------

    def _tick_cover(self) -> None:
        """Advance the IGBNN inner loop by one tick over all robots."""
        for i, robot in enumerate(self.robots):
            if i in self._idle:
                continue                 # idles through this whole minigraph
            deadlocked = self._trace[i][0] == self._trace[i][8]

            if robot[5] != robot[6] and not deadlocked:
                leader = self._robot_by_id(robot[5])
                # Adjacency guard: a follower must stay a direct neighbour of
                # its host.  Neuron skipping lets a host jump several cells
                # at once, so without this the pair silently drifts apart and
                # the follower keeps mirroring a motion vector that no longer
                # describes its own surroundings.
                if leader is None or max(abs(robot[0] - leader[0]),
                                         abs(robot[1] - leader[1])) > 1:
                    self._detach(i)
                    self._step_autonomous(i)
                else:
                    nx, ny = robot[0] + leader[3], robot[1] + leader[4]
                    if (self._valid(nx, ny)
                            and (nx, ny) not in set(self._robot_locals())):
                        self._commit_move(i, (nx, ny))
                    else:
                        self._detach(i)
                        self._step_autonomous(i)
            else:
                if deadlocked and robot[5] != robot[6]:
                    self._detach(i)
                self._step_autonomous(i)

            self._trace[i] = [(self.robots[i][0], self.robots[i][1])] \
                + self._trace[i][:9]

        self._assert_no_overlap("cover")

        # No-progress detector.  A step cap alone is a blunt instrument: it
        # is either loose enough to waste tens of thousands of ticks before
        # firing, or tight enough to truncate a legitimately long sweep.
        # Counting ticks since the last newly-covered cell separates the two
        # cases directly -- a healthy sweep resets this constantly, a stuck
        # one trips it within a couple of hundred ticks.
        h, v = self._mini_shape
        remaining = int(np.sum(
            self.state[self._ro:self._ro + h, self._co:self._co + v]
            == UNVISITED))
        if remaining < self._last_remaining:
            self._stale_ticks = 0
            self._last_remaining = remaining
        else:
            self._stale_ticks += 1

        self._refresh_bias_field()
        self._update_activity()
        self._mini_step += 1

    def _step_autonomous(self, idx: int) -> None:
        self._xfm(idx)
        nbs = self._nav_nbrs(self.robots[idx])
        self._commit_move(idx, self._next_wp(idx, nbs))

    def _assert_no_overlap(self, where: str) -> None:
        pos = [(r[0], r[1]) for r in self.robots]
        if len(pos) != len(set(pos)):
            dupes = sorted({p for p in pos if pos.count(p) > 1})
            raise RuntimeError(
                f"[IGBNN-mu] two robots occupy the same cell during {where} "
                f"at tick {self._step}: positions={pos} duplicates={dupes}")

    # ---- transit setup / tick ----------------------------------------

    def _begin_transit(self, targets: Sequence[Cell]) -> None:
        """Plan A* paths from the robots' current GLOBAL poses to `targets`."""
        # Section IV-C: the next minigraph may seat fewer robots than we
        # have.  Whoever has no entry cell simply holds position and idles
        # through that minigraph, so pad the target list with each surplus
        # robot's own pose rather than indexing off the end of it.
        targets = list(targets)
        if len(targets) < self.n:
            self._warn(
                f"next minigraph seats only {len(targets)} of {self.n} "
                f"robots; the remainder hold position (Section IV-C)")
            claimed = set(targets)
            for r in self.robots[len(targets):]:
                cell = (r[0], r[1])
                while cell in claimed:
                    cell = (cell[0], cell[1] + 1)
                claimed.add(cell)
                targets.append((r[0], r[1]))

        self._transit_targets = list(targets)
        self._transit_paths = []
        for i, r in enumerate(self.robots):
            start = (r[0], r[1])
            path = self._astar(start, targets[i])
            if path is None:
                # The assigned entry cell is walled off from this robot.
                # Fall back to the nearest cell we can actually reach rather
                # than teleporting, which the original did silently.
                alt = self._nearest_reachable(
                    start, self._reachable_entry_alternatives(targets[i]))
                if alt is None:
                    self._warn(
                        f"robot {r[6]} cannot reach any entry cell of "
                        f"minigraph {self._order[self._visit_idx + 1]}; "
                        f"it holds position at {start}")
                    path = [start]
                else:
                    self._transit_targets[i] = alt
                    path = self._astar(start, alt) or [start]
            self._transit_paths.append(path)
        self._transit_ptrs = [1] * self.n
        self._transit_blocked = [0] * self.n
        self._transit_step = 0
        self._phase = "transit"

    def _reachable_entry_alternatives(self, around: Cell) -> List[Cell]:
        """Free cells near `around`, nearest first -- fallback entry points."""
        cands = [(gx, gy)
                 for gx in range(self.H) for gy in range(self.V)
                 if self.state[gx, gy] != OBSTACLE]
        cands.sort(key=lambda c: (c[0] - around[0]) ** 2
                   + (c[1] - around[1]) ** 2)
        return cands[:64]

    #: Ticks a robot may sit blocked in transit before its path is replanned
    #: around whatever is in the way.
    TRANSIT_REPLAN_AFTER = 4

    def _tick_transit(self) -> bool:
        """Advance every robot one cell along its transit path.

        Naive priority ordering is not enough here.  Two robots whose next
        cells are each other's current cell -- a straight swap -- block each
        other forever: neither cell is ever free at the moment the other
        robot is served, so both sit still until the step cap fires.  This
        was the single largest source of wasted transit ticks (measured:
        500 ticks burned and both robots left stranded, on a plain 30x30
        map at 5% obstacle density).

        Resolution is in two passes per tick:

          1. *Chain moves* -- repeatedly let any robot whose next cell is
             genuinely unoccupied move, so a robot that vacates a cell frees
             the robot queued behind it within the same tick.
          2. *Cycle moves* -- among whoever is still stuck, find directed
             cycles in the "wants the cell occupied by" graph and step every
             member of a cycle simultaneously.  A 2-cycle is a swap; longer
             cycles are rotations around a loop.  Because the whole cycle
             moves at once, no cell is ever doubly occupied at a tick
             boundary.

        Anything still blocked after both passes is genuinely obstructed by
        a robot that has parked on its target, and gets its path replanned
        around the parked robots after TRANSIT_REPLAN_AFTER ticks.

        Returns True once every robot has arrived (or has been shown to have
        no route left, in which case it holds and a warning is recorded).
        """
        n = self.n
        arrived = [self._transit_ptrs[i] >= len(self._transit_paths[i])
                   for i in range(n)]
        desired: Dict[int, Cell] = {
            i: self._transit_paths[i][self._transit_ptrs[i]]
            for i in range(n) if not arrived[i]
        }
        occupant: Dict[Cell, int] = {
            (r[0], r[1]): i for i, r in enumerate(self.robots)}
        moved: set = set()

        # ---- pass 1: chain moves --------------------------------------
        progress = True
        while progress:
            progress = False
            for i in range(n):
                if i in moved or i not in desired:
                    continue
                cell = desired[i]
                if cell in occupant:
                    continue
                del occupant[(self.robots[i][0], self.robots[i][1])]
                occupant[cell] = i
                self._apply_transit_move(i, cell)
                moved.add(i)
                progress = True

        # ---- pass 2: cycle moves --------------------------------------
        stuck = [i for i in desired if i not in moved]
        succ = {i: occupant[desired[i]]
                for i in stuck if desired[i] in occupant}
        for cycle in self._find_cycles(succ):
            if any(j in moved for j in cycle):
                continue
            # Snapshot targets first, then commit, so each robot in the
            # cycle steps onto the cell its predecessor is vacating.
            targets = {j: desired[j] for j in cycle}
            for j, cell in targets.items():
                self._apply_transit_move(j, cell)
                moved.add(j)

        # ---- blocked bookkeeping --------------------------------------
        for i in range(n):
            if arrived[i]:
                continue
            if i in moved:
                self._transit_blocked[i] = 0
            else:
                self._transit_blocked[i] += 1
                if self._transit_blocked[i] >= self.TRANSIT_REPLAN_AFTER:
                    self._replan_transit(i)

        self._assert_no_overlap("transit")
        self._transit_step += 1
        return all(self._transit_ptrs[i] >= len(self._transit_paths[i])
                   for i in range(n))

    def _apply_transit_move(self, i: int, cell: Cell) -> None:
        """Commit one transit step for robot `i` onto global `cell`."""
        r = self.robots[i]
        r[3], r[4] = cell[0] - r[0], cell[1] - r[1]
        r[0], r[1] = cell
        if self.state[cell] != OBSTACLE:
            self.state[cell] = VISITED
        self.path_plans[r[6]].append(cell)
        self._transit_ptrs[i] += 1

    @staticmethod
    def _find_cycles(succ: Dict[int, int]) -> List[List[int]]:
        """All simple cycles in the functional graph `succ` (i -> succ[i]).

        Each node has at most one outgoing edge, so every cycle is disjoint
        and a single walk per node finds them all in linear time.
        """
        cycles: List[List[int]] = []
        seen: set = set()
        for start in succ:
            if start in seen:
                continue
            walk: List[int] = []
            index: Dict[int, int] = {}
            node = start
            while node in succ and node not in seen:
                if node in index:
                    cycles.append(walk[index[node]:])
                    break
                index[node] = len(walk)
                walk.append(node)
                node = succ[node]
            seen.update(walk)
        return cycles

    def _replan_transit(self, i: int) -> None:
        """Re-route robot `i` around the robots that are parked on targets."""
        r = self.robots[i]
        start = (r[0], r[1])
        parked = {(o[0], o[1]) for j, o in enumerate(self.robots)
                  if j != i and self._transit_ptrs[j] >= len(
                      self._transit_paths[j])}
        target = self._transit_targets[i]
        path = self._astar(start, target, blocked=parked)

        if path is None:
            # Every route to the assigned entry cell runs through a robot
            # that has parked and will never move.  Re-planning to the same
            # target is then futile -- the earlier version fell back to the
            # unblocked path, which is exactly the one that is permanently
            # obstructed, and the robot span until the step cap.  Take a
            # different entry cell instead: the nearest free cell in the
            # destination minigraph that is genuinely reachable right now.
            alt = self._alternative_entry(i, start, target, parked)
            if alt is None:
                self._warn(
                    f"robot {r[6]} has no route to entry cell {target} "
                    f"and no reachable alternative; holding at {start}")
                self._transit_paths[i] = [start]
                self._transit_ptrs[i] = 1
                self._transit_blocked[i] = 0
                return
            self._transit_targets[i] = alt
            path = self._astar(start, alt, blocked=parked) or [start]

        self._transit_paths[i] = path
        self._transit_ptrs[i] = 1
        self._transit_blocked[i] = 0

    def _alternative_entry(self, i: int, start: Cell, target: Cell,
                           parked: set) -> Optional[Cell]:
        """Nearest reachable free cell in the destination minigraph.

        Excludes cells already assigned to another robot so two robots never
        converge on the same entry position.
        """
        if self._visit_idx + 1 >= len(self._order):
            return None
        nxt = self._order[self._visit_idx + 1]
        ro, co = self._minis[nxt[0] * self._nc + nxt[1]]
        h, v = self._mini_shape
        claimed = {t for j, t in enumerate(self._transit_targets) if j != i}

        cands = [(gx, gy)
                 for gx in range(ro, ro + h)
                 for gy in range(co, co + v)
                 if self.state[gx, gy] != OBSTACLE
                 and (gx, gy) not in claimed
                 and (gx, gy) not in parked]
        cands.sort(key=lambda c: (c[0] - target[0]) ** 2
                   + (c[1] - target[1]) ** 2)
        for c in cands:
            if self._astar(start, c, blocked=parked) is not None:
                return c
        return None

    def _warn(self, msg: str) -> None:
        self._warnings.append(msg)

    # ---- mop-up ------------------------------------------------------

    def _build_mopup_queue(self) -> None:
        """Free cells still unvisited but reachable from a robot.

        The serpentine sweep covers each minigraph in isolation, so a region
        whose only connection to the rest of the map runs *through* another
        minigraph gets skipped by the per-minigraph reachability test.  This
        pass catches exactly those cells, which is what the completeness
        proof in thesis Appendix A requires.
        """
        seeds = [(r[0], r[1]) for r in self.robots]
        reach = self._flood(seeds)
        remaining = np.argwhere((self.state == UNVISITED) & reach)
        self._mopup_queue = [tuple(c) for c in remaining]

    def _tick_mopup(self) -> bool:
        """Send robot 0 to the next outstanding cell.  True when queue empty."""
        while self._mopup_queue:
            target = self._mopup_queue.pop(0)
            if self.state[target] != UNVISITED:
                continue                       # covered en route already
            r = self.robots[0]
            path = self._astar((r[0], r[1]), target)
            if path is None:
                continue
            for (gx, gy) in path[1:]:
                r[0], r[1] = gx, gy
                if self.state[gx, gy] != OBSTACLE:
                    self.state[gx, gy] = VISITED
                self.path_plans[r[6]].append((gx, gy))
                self._step += 1
            return False
        return True

    # ---- the public tick ---------------------------------------------

    def step(self) -> bool:
        """Advance the planner by one tick.

        Returns True while work remains (call again), False once the planner
        has terminated.  Subsequent calls after termination are no-ops that
        keep returning False.
        """
        if not self._started:
            raise RuntimeError(
                "IGBNN_mu.step() called before reset(); call reset() first.")
        if self._done:
            return False

        self._step += 1

        if self._phase == "cover":
            self._tick_cover()
            h, v = self._mini_shape
            stale_limit = max(200, 2 * h * v)
            capped = self._mini_step >= self.mini_step_cap
            stalled = self._stale_ticks >= stale_limit
            if capped:
                self._n_mini_capped += 1
                self._warn(
                    f"minigraph {self._order[self._visit_idx]} hit "
                    f"mini_step_cap={self.mini_step_cap}")
            elif stalled:
                self._n_mini_stalled += 1
                self._warn(
                    f"minigraph {self._order[self._visit_idx]} made no "
                    f"coverage progress for {stale_limit} ticks with "
                    f"{self._last_remaining} cell(s) left; moving on (the "
                    f"mop-up pass will retry them from another approach)")
            capped = capped or stalled
            if not self._mini_incomplete() or capped:
                self._advance_minigraph()

        elif self._phase == "transit":
            arrived = self._tick_transit()
            if self._transit_step >= self.transit_step_cap and not arrived:
                self._n_transit_capped += 1
                self._warn(
                    f"transit into {self._order[self._visit_idx + 1]} hit "
                    f"transit_step_cap={self.transit_step_cap}; robots "
                    f"continue from wherever they stalled")
                arrived = True
            if arrived:
                self._begin_minigraph(self._visit_idx + 1)

        elif self._phase == "mopup":
            if self._tick_mopup():
                self._finish("complete")

        if self.visualize and self._step % self.viz_interval == 0:
            self._viz(self._phase)

        return not self._done

    def _advance_minigraph(self) -> None:
        """Leave the current minigraph: transit to the next, or finish."""
        # Local -> global.
        for r in self.robots:
            r[0] += self._ro
            r[1] += self._co

        if self._visit_idx >= len(self._order) - 1:
            if self.mop_up:
                self._build_mopup_queue()
                if self._mopup_queue:
                    self._phase = "mopup"
                    return
            self._finish("complete")
            return

        nxt = self._order[self._visit_idx + 1]
        nro, nco = self._minis[nxt[0] * self._nc + nxt[1]]
        edge = self._entry_dir(self._order[self._visit_idx], nxt)
        self._begin_transit(self._entry_targets(nro, nco, edge))

    def _finish(self, reason: str) -> None:
        self._done = True
        self._stop_reason = reason
        self._phase = "done"

    def is_done(self) -> bool:
        return bool(self._done)

    def final_result(self) -> Tuple[Dict[int, List[Cell]], Dict[str, Any]]:
        """(paths, stats) after termination.  Idempotent."""
        if not self._started:
            raise RuntimeError("IGBNN_mu.final_result() called before reset().")

        free_total = int(np.sum(self.full_grid == UNVISITED))
        remaining = int(np.sum(self.state == UNVISITED))
        covered = free_total - remaining

        # Which of the leftovers were ever reachable at all?
        reach = self._flood([(r[0], r[1]) for r in self.robots])
        unreachable = int(np.sum((self.state == UNVISITED) & ~reach))

        stats = {
            "steps": self._step,
            "coverage": covered / max(free_total, 1),
            "free_cells": free_total,
            "covered_cells": covered,
            "remaining_cells": remaining,
            "unreachable_cells": unreachable,
            "minigraphs": len(self._minis),
            "mini_shape": self._mini_shape,
            "path_len": {rid: len(p) for rid, p in self.path_plans.items()},
            "wall_time": time.time() - self._t0,
            "stop_reason": self._stop_reason or "running",
            "mini_step_capped": self._n_mini_capped,
            "mini_stalled": self._n_mini_stalled,
            "minis_skipped_blocked": self._n_minis_skipped,
            "idle_assignments": self._n_idle_assignments,
            "orig_shape": self.orig_shape,
            "padded_shape": (self.H, self.V),
            "decomposition": self.decomposition,
            "transit_step_capped": self._n_transit_capped,
            "warnings": list(self._warnings),
        }
        return self.path_plans, stats

    def render_state(self) -> Dict[str, Any]:
        """Snapshot of live planner state for external renderers.

        Fresh dict per call; arrays are defensive copies so a caller can hold
        one across ticks without aliasing the planner's own buffers.
        """
        h, v = self._mini_shape
        return {
            "shape": (self.H, self.V),
            "state": self.state.copy(),
            "activity": self._act.copy(),
            "mini_origin": (self._ro, self._co),
            "mini_shape": self._mini_shape,
            "mini_index": self._visit_idx,
            "mini_total": len(self._order),
            "mini_rc": self._order[self._visit_idx] if self._order else None,
            "phase": self._phase,
            "robots": [
                {
                    "id": r[6],
                    "pos": (r[0] + (self._ro if self._phase == "cover" else 0),
                            r[1] + (self._co if self._phase == "cover" else 0)),
                    "local": (r[0], r[1]),
                    "heading": (r[3], r[4]),
                    "leader": r[5],
                    "fused_v": list(r[7]),
                    "fused_h": list(r[8]),
                }
                for r in self.robots
            ],
            "paths": {rid: list(p) for rid, p in self.path_plans.items()},
            "step": self._step,
            "done": self.is_done(),
            "stop_reason": self._stop_reason,
        }

    def robot_global_positions(self) -> List[Cell]:
        """Robot positions in GLOBAL coordinates regardless of phase."""
        if self._phase == "cover":
            return [(r[0] + self._ro, r[1] + self._co) for r in self.robots]
        return [(r[0], r[1]) for r in self.robots]

    # ==================================================================
    #  SECTION 9 -- BLOCKING run()
    # ==================================================================

    def run(self, max_steps: Optional[int] = None
            ) -> Tuple[Dict[int, List[Cell]], Dict[str, Any]]:
        """Algorithm 4, blocking.

        1. Decompose the map into minigraphs (Rule 1 + Rule 2).
        2. Visit them in serpentine order.  For each: run the IGBNN inner
           loop (Algorithm 2), then A*-transit to the next entry edge.
        3. Mop up any reachable cell the decomposition stranded.

        Returns (path_plans, stats).
        """
        self.reset()
        if self.verbose:
            print(f"[IGBNN-mu] Grid {self.H}x{self.V} | {self.n} robots")
            print(f"[IGBNN-mu] Decomposition: {self._decomp_note}")

        cap = max_steps if max_steps is not None else 10 ** 7
        while self.step():
            if self._step >= cap:
                self._finish("max_steps")
                break

        paths, stats = self.final_result()
        if self.verbose:
            print(f"[IGBNN-mu] {stats['stop_reason']}: "
                  f"coverage={stats['coverage']:.4f} "
                  f"({stats['covered_cells']}/{stats['free_cells']}) "
                  f"steps={stats['steps']}")
            for w in stats["warnings"]:
                print(f"[IGBNN-mu]   warning: {w}")
        return paths, stats

    verbose: bool = True

    # ==================================================================
    #  SECTION 10 -- VISUALISATION  (matplotlib imported lazily)
    # ==================================================================
    #
    #  Two render paths, mirroring Interstar's split:
    #
    #    _viz()  live, per-iteration.  Reuses one persistent named figure,
    #            redraws in place, and returns immediately.  Driven from
    #            reset() and step() whenever `visualize` is on.
    #    plot()  final, blocking.  Fresh figure, plt.show().
    #
    #  The live path must not call plt.show(): show() blocks until the
    #  window is closed, so a 447-step coverage run would demand 447 manual
    #  closes.  It must also not call plt.subplots() per frame -- that
    #  leaks a figure per iteration and trips matplotlib's 20-figure
    #  warning within seconds.  Hence plt.ion() + a figure looked up by
    #  name + fig.clf() + draw_idle() + pause(), which is exactly what
    #  Interstar._viz does.

    #: Window title of the persistent live figure.
    _VIZ_WINDOW = "IGBNN-mu coverage"

    def _state_rgb(self) -> "np.ndarray":
        """(H, V, 3) uint8 image of the coverage state.

        Drawn with imshow rather than one Rectangle patch per cell: a
        30x30 map is 900 patches per frame, which is slow enough to
        dominate the run once every iteration is rendered.
        """
        img = np.empty((self.H, self.V, 3), dtype=np.uint8)
        img[...] = (0x5F, 0x74, 0x72)                       # visited
        img[self.state == UNVISITED] = (0xC1, 0xC1, 0xC1)   # unvisited
        img[self.state == OBSTACLE] = (0x0A, 0x09, 0x0A)    # obstacle
        return img

    def combined_units(self) -> Dict[int, List[int]]:
        """Group robot indices into physically combined units.

        Section III-C: "the attached robot with a greater index number will
        follow the host robot's direction... The combined robot identifies
        its index with the robot with the smallest index."  A robot with
        alpha != beta is attached to the host named by alpha, and a host can
        itself be attached, so the chain is followed to its root.

        Returns {root_index: [member indices]}, singletons included.  The
        root is the lowest-indexed member, i.e. the index the combined unit
        takes as its own.
        """
        idx_of = {r[6]: i for i, r in enumerate(self.robots)}
        units: Dict[int, List[int]] = {}
        for i in range(len(self.robots)):
            cur, seen = i, set()
            while cur not in seen:
                seen.add(cur)
                row = self.robots[cur]
                if row[5] == row[6]:
                    break                      # independent: this is the root
                nxt = idx_of.get(row[5])
                if nxt is None:
                    break
                cur = nxt
            units.setdefault(cur, []).append(i)
        return units

    @staticmethod
    def outline_segments(cells: Sequence[Cell]) -> List[Tuple[Cell, Cell]]:
        """Silhouette of a set of grid cells, in CORNER coordinates.

        An edge of a cell is on the boundary exactly when the cell across
        it is not part of the set, so collecting those edges traces the
        outline of the combined footprint — concave shapes included, which
        matters because an L-shaped or staggered formation is legal.

        Cell (r, c) spans corner rows r..r+1 and corner columns c..c+1.
        Returned segments are ((col0, row0), (col1, row1)) in that corner
        frame; callers map them into whatever space they draw in.
        """
        s = set(cells)
        segs: List[Tuple[Cell, Cell]] = []
        for (r, c) in s:
            if (r - 1, c) not in s:
                segs.append(((c, r), (c + 1, r)))
            if (r + 1, c) not in s:
                segs.append(((c, r + 1), (c + 1, r + 1)))
            if (r, c - 1) not in s:
                segs.append(((c, r), (c, r + 1)))
            if (r, c + 1) not in s:
                segs.append(((c + 1, r), (c + 1, r + 1)))
        return segs

    def _draw_axes(self, ax, label: str) -> None:
        """Paint one frame of the coverage state onto `ax`."""
        ax.imshow(self._state_rgb(), origin="upper", interpolation="nearest")

        # Minigraph lattice.
        h, v = self._mini_shape
        if h and v:
            for r in range(0, self.H + 1, h):
                ax.axhline(r - 0.5, color="#E8B21A", linewidth=0.6, alpha=0.55)
            for c in range(0, self.V + 1, v):
                ax.axvline(c - 0.5, color="#E8B21A", linewidth=0.6, alpha=0.55)
            # Active tile.
            if self._started:
                import matplotlib.patches as patches
                ax.add_patch(patches.Rectangle(
                    (self._co - 0.5, self._ro - 0.5), v, h,
                    fill=False, edgecolor="#E8B21A", linewidth=2.0))

        pos = self.robot_global_positions()

        # Combined-unit silhouette.  Two robots that have inter-reconfigured
        # are one rigid body planning as a single agent (Section III-C), and
        # the enlarged footprint is what neuron skipping is measured against
        # — so it is drawn as one outline rather than left as loose dots.
        # Only during the cover phase: fusion state is re-formed per
        # minigraph, so between tiles it is stale.
        if self._started and self._phase == "cover":
            from matplotlib.collections import LineCollection
            for root, members in self.combined_units().items():
                if len(members) < 2:
                    continue
                col = self.COLORS[root % len(self.COLORS)]
                segs = [((c0 - 0.5, r0 - 0.5), (c1 - 0.5, r1 - 0.5))
                        for (c0, r0), (c1, r1) in
                        self.outline_segments([pos[i] for i in members])]
                # White casing under the coloured line.  The outline is
                # drawn in the host's colour to say which unit it is, but
                # that colour is also worn by one of the members and can
                # land on either the light unvisited or dark visited
                # shade, so on its own it reads poorly in both directions.
                ax.add_collection(LineCollection(
                    segs, colors="white", linewidths=5.0, zorder=6))
                ax.add_collection(LineCollection(
                    segs, colors=col, linewidths=2.4, zorder=7))

        # Robots.  Idle robots (Section IV-C) are drawn hollow so it is
        # visible at a glance which ones this minigraph could not seat.
        for idx, (gx, gy) in enumerate(pos):
            col = self.COLORS[idx % len(self.COLORS)]
            idle = idx in self._idle
            ax.scatter(gy, gx, c="white", s=220, zorder=8)
            ax.scatter(gy, gx, c="none" if idle else col, s=150, zorder=9,
                       edgecolors=col, linewidths=1.8)

        ax.set_xlim(-0.5, self.V - 0.5)
        ax.set_ylim(self.H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"IGBNN-mu  |  {label}  |  step {self._step}",
                     fontsize=13)

    def _viz(self, label: str = "") -> None:
        """Live in-place redraw of one iteration.  Non-blocking.

        matplotlib is imported here rather than at module scope so that
        importing this module on a headless box costs nothing and pulls in
        no GUI stack.  Same rule the GBNN+H module follows.
        """
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure(num=self._VIZ_WINDOW, figsize=(7, 7))
        fig.clf()
        ax = fig.add_subplot(111)
        self._draw_axes(ax, label)
        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(self.viz_pause)

    def plot(self, title: Optional[str] = None) -> None:
        """Render the current coverage state and block until dismissed.

        Counterpart to ``Interstar.visualize()``.  Named ``plot`` rather
        than ``visualize`` because ``visualize`` is already the constructor
        flag on this class — as it is on ``GBNN_H`` — and shadowing it would
        make ``planner.visualize()`` a call on a bool.

        matplotlib is imported lazily, so this costs nothing on a headless
        machine until it is actually called.
        """
        import matplotlib.pyplot as plt

        plt.ioff()
        fig, ax = plt.subplots(figsize=(8, 8))
        self._draw_axes(ax, title if title is not None else "complete")
        fig.tight_layout()
        plt.show()
# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import time

    # ── Coverage: 30x30 random map, 3 robots, rendered every iteration ──────
    #
    # visualize=True with viz_interval=1 redraws the persistent
    # "IGBNN-mu coverage" window on every step, so the sweep, the minigraph
    # hand-offs and the A* transits between them are all watchable live.
    # Set visualize=False for a headless timing run.
    grid    = make_grid(30, 30, obstacle_chance=0.05, seed=42)
    planner = IGBNN_mu(grid, n_robots=3,
                       visualize=True, viz_interval=1, viz_pause=0.02)
    t0      = time.time()
    paths, stats = planner.run()
    print(f"Coverage ({time.time()-t0:.3f}s)")
    for rid, p in sorted(paths.items()):
        print(f"  Robot {rid:2d}: {len(p):4d} wp  {p[0]} → {p[-1]}")
    planner.plot()

    # ── Minigraphs: same map, decomposition reported ────────────────────────
    print(f"\nMinigraphs")
    print(f"  grid        {stats['orig_shape'][0]}x{stats['orig_shape'][1]}"
          f"  → padded {stats['padded_shape'][0]}x{stats['padded_shape'][1]}")
    print(f"  partition   {planner._nr}x{planner._nc} tiles of "
          f"{stats['mini_shape'][0]}x{stats['mini_shape'][1]}"
          f"  ({stats['minigraphs']} total)")
    print(f"  coverage    {stats['coverage']*100:.2f}%  "
          f"({stats['covered_cells']}/{stats['free_cells']} cells)"
          f"  in {stats['steps']} steps")
    for w in stats['warnings']:
        print(f"  note        {w}")

    # ── Scalability: effort per sweep falls as the team grows ───────────────
    print(f"\nScalability (REF_GRIDS[1], 20x20 with two pillars)")
    for n in (1, 2, 3, 4, 5):
        sim  = IGBNN_mu(REF_GRIDS[1].copy(), n_robots=n)  # headless
        sim.verbose = False
        t0   = time.time()
        _, st = sim.run()
        print(f"  n={n}: {st['steps']:4d} steps  "
              f"{st['coverage']*100:6.2f}% coverage  "
              f"({time.time()-t0:.3f}s)")
