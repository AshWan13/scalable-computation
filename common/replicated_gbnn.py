#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replicated_gbnn.py — Reference implementation of the Glasius Bioinspired
                     Neural Network (GBNN) base coverage planner.

ATTRIBUTION
-----------
This module is a reference re-implementation of the GBNN activity-propagation
algorithm originally proposed by:

    Glasius, R., Komoda, A., & Gielen, S. C. A. M. (1995).
    "Neural network dynamics for path planning and obstacle avoidance."
    Neural Networks, 8(1), 125–133.

GBNN itself is NOT a contribution of this codebase. It is included here only
as the prior-art baseline upon which the derivative algorithms in this
repository (e.g. GBNN+H) build. All credit for the underlying neural
dynamics belongs to Glasius et al. (1995).

DESCRIPTION
-----------
Pure, standalone implementation of the GBNN activity-propagation algorithm
(Eqns 2–4) for single-unit area coverage. Applicable to both split singletons
and fused singletons — morphology is implicit in the grid cell size handed
to reset(). Derivative algorithms with neuron skipping, minigraphs, and
multi-robot coordination live in their own modules, not here.

The class is frame-driven: one step() call = one GBNN iteration. Dynamic
obstacles (doors, humans, other robots) can be pushed in-place via
set_occupancy() between steps without losing coverage progress.

EQUATIONS
---------
    Eqn 2 (activity fn):    f(x) = -1      if x < 0
                                   b·x      if 0 ≤ x < 1
                                   +1       if x ≥ 1

    Eqn 3 (neuron update):  n_ij(t+1) = f(  Σ_j w_ij · max(n_j, 0)  +  I_i )
                            I_i =  +E   on unvisited cell (+1)
                                   -E   on obstacle      (-1)
                                    0   on visited / residual

    Eqn 4 (weight):         w_ij = exp(-α · d²)  if d < r
                                   0             otherwise
                            (d = Euclidean distance between cells i and j)

GRID ENCODING
-------------
    +1.0   unvisited (drives attraction)
    -1.0   obstacle  (repels)
    other  visited / transient activity residue
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


class GBNN:
    """Glasius Bioinspired Neural Network base coverage planner.

    One unit (split or fused singleton) traverses a 2D grid, marking cells
    covered as it moves. Output is a sequence of grid-cell waypoints;
    kinematics (holonomic, diff-drive, etc.) are enforced by the caller.

    Parameters
    ----------
    E : float, default 100.0
        External input magnitude (±E on unvisited/obstacle cells).
    r : float, default 2.0
        Receptive-field radius in cell units (Eqn 4).
    alpha : float, default 2.0
        Weight decay coefficient (Eqn 4).
    b : float, default 0.7
        Activity-function slope for 0 ≤ x < 1 (Eqn 2).
    c : float, default 0.01
        Cost-distance coefficient in the greedy next-cell selector.

    Typical usage
    -------------
        gbnn = GBNN()
        gbnn.reset(grid, start=(row, col))
        while not gbnn.is_done():
            gbnn.set_occupancy(live_obstacle_mask)   # optional per-tick
            pos = gbnn.step()
            # drive the robot to pos; when it arrives, loop
    """

    # ---- paper defaults (overridable via __init__) ----
    E_DEFAULT:     float = 100.0
    R_DEFAULT:     float = 2.0
    ALPHA_DEFAULT: float = 2.0
    B_DEFAULT:     float = 0.7
    C_DEFAULT:     float = 0.01

    def __init__(
        self,
        E: float = E_DEFAULT,
        r: float = R_DEFAULT,
        alpha: float = ALPHA_DEFAULT,
        b: float = B_DEFAULT,
        c: float = C_DEFAULT,
    ) -> None:
        self.E     = float(E)
        self.r     = float(r)
        self.alpha = float(alpha)
        self.b     = float(b)
        self.c     = float(c)

        # Reset-initialised state
        self._grid:       Optional[np.ndarray] = None
        self._xi:         Optional[List[int]]  = None   # [row, col]  current
        self._xp:         Optional[List[int]]  = None   # [row, col]  previous
        self._iterations: int                  = 0
        self._path:       List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
    ) -> None:
        """Initialise with a fresh grid and a starting cell.

        Parameters
        ----------
        grid : (H, W) array-like
            +1.0 unvisited, -1.0 obstacle, anything else visited/residue.
        start : (row, col)
            Starting cell; must be non-obstacle.
        """
        g = np.array(grid, dtype=float, copy=True)
        if g.ndim != 2:
            raise ValueError(f"grid must be 2D, got shape {g.shape}")
        sr, sc = int(start[0]), int(start[1])
        H, W = g.shape
        if not (0 <= sr < H and 0 <= sc < W):
            raise ValueError(f"start {start} out of grid bounds {g.shape}")
        if g[sr, sc] == -1.0:
            raise ValueError(f"start {start} is an obstacle cell")

        self._grid       = g
        self._xi         = [sr, sc]
        self._xp         = [sr, sc + 1]   # arbitrary pseudo-previous
        self._iterations = 0
        self._path       = [(sr, sc)]

        # Mark start cell covered (smooth residue rather than raw 0)
        self._grid[sr, sc] = self._update_xi_to_covered((sr, sc))

    def set_occupancy(self, mask: np.ndarray) -> None:
        """Overwrite obstacle cells from a live occupancy mask.

        Cells where mask is True become -1 (obstacle). Cells that were
        formerly obstacle but are now clear are reset to +1 (unvisited) so
        activity propagation re-attracts the unit. Non-obstacle cells that
        have been visited keep their residual activity — coverage progress
        is preserved across dynamic obstacle updates.

        Call this between steps, as often as every step, to track moving
        humans / opening doors / other robots without re-planning.
        """
        if self._grid is None:
            return
        m = np.asarray(mask, dtype=bool)
        if m.shape != self._grid.shape:
            raise ValueError(
                f"mask shape {m.shape} must match grid {self._grid.shape}"
            )

        was_obstacle = (self._grid == -1.0)
        # Newly blocked cells → -1
        self._grid[m] = -1.0
        # Formerly-obstacle, now-clear cells → +1 (re-activate as unvisited)
        cleared = was_obstacle & (~m)
        self._grid[cleared] = 1.0

    def step(self) -> Tuple[int, int]:
        """One GBNN iteration.

        1. Propagate activity across the whole grid via Eqn 3.
        2. Pick the highest-scoring free neighbour (greedy).
        3. Move xi to that neighbour and mark the cell covered.

        Returns
        -------
        (row, col) : the new unit cell position.
            If the run is already complete or the unit is isolated, returns
            the current cell unchanged.
        """
        if self._grid is None or self._xi is None:
            raise RuntimeError("call reset(grid, start) before step()")
        if self.is_done():
            return tuple(self._xi)

        # 1. Activity propagation over whole grid (Eqn 3)
        self._grid = self._update_grid()

        # 2. Greedy next cell
        neighbours = self._get_neighbours(self._xi)
        if not neighbours:
            return tuple(self._xi)
        next_xi = self._next_position(self._xi, self._xp, neighbours)

        # 3. Move + mark covered
        self._xp = list(self._xi)
        self._xi = list(next_xi)
        if self._grid[self._xi[0], self._xi[1]] != -1.0:
            self._grid[self._xi[0], self._xi[1]] = \
                self._update_xi_to_covered(self._xi)
        self._path.append(tuple(self._xi))
        self._iterations += 1
        return tuple(self._xi)

    def is_done(self) -> bool:
        """True when no unvisited (+1.0) cells remain."""
        if self._grid is None:
            return False
        return not bool(np.any(self._grid == 1.0))

    # ------------------------------------------------------------------
    #  Read-only properties (for demo.py HUD + heatmap overlay)
    # ------------------------------------------------------------------

    @property
    def activity_grid(self) -> Optional[np.ndarray]:
        """Copy of the current neuron-activity grid."""
        return None if self._grid is None else self._grid.copy()

    @property
    def coverage_pct(self) -> float:
        """Fraction of free cells covered, in [0, 1]."""
        if self._grid is None:
            return 0.0
        n_free  = int((self._grid != -1.0).sum())
        n_unvis = int((self._grid == 1.0).sum())
        if n_free == 0:
            return 0.0
        return (n_free - n_unvis) / n_free

    @property
    def position(self) -> Optional[Tuple[int, int]]:
        return None if self._xi is None else (int(self._xi[0]), int(self._xi[1]))

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def path(self) -> List[Tuple[int, int]]:
        """Accumulated trajectory as list of (row, col)."""
        return list(self._path)

    # ------------------------------------------------------------------
    #  Core math (Eqns 2, 3, 4) — private
    # ------------------------------------------------------------------

    def _w(self, xi, xj) -> float:
        """Eqn 4: weight coefficient w_ij."""
        d = math.hypot(xi[0] - xj[0], xi[1] - xj[1])
        return math.exp(-self.alpha * d * d) if d < self.r else 0.0

    def _G(self, x: float) -> float:
        """Eqn 2: activity function f(x)."""
        if x < 0.0:
            return -1.0
        if x >= 1.0:
            return 1.0
        return self.b * x

    def _neuron(self, xi) -> float:
        """Eqn 3: neuron update at cell xi."""
        s = 0.0
        for j in self._get_neighbours(xi):
            s += self._w(xi, j) * max(float(self._grid[j[0], j[1]]), 0.0)
        val = float(self._grid[xi[0], xi[1]])
        if val == 1.0:
            Ii = self.E
        elif val == -1.0:
            Ii = -self.E
        else:
            Ii = 0.0
        return self._G(s + Ii)

    def _update_grid(self) -> np.ndarray:
        """Apply Eqn 3 over every cell; return the new grid."""
        new = self._grid.copy()
        H, W = self._grid.shape
        for i in range(H):
            for j in range(W):
                new[i, j] = self._neuron((i, j))
        return new

    def _update_xi_to_covered(self, xi) -> float:
        """Value written into the just-covered cell (smooth residue)."""
        s = 0.0
        for j in self._get_neighbours(xi):
            s += self._w(xi, j) * max(float(self._grid[j[0], j[1]]), 0.0)
        return s

    def _get_neighbours(self, xi) -> List[Tuple[int, int]]:
        """8-way connected free neighbours of xi."""
        out: List[Tuple[int, int]] = []
        H, W = self._grid.shape
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = xi[0] + dr, xi[1] + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if float(self._grid[nr, nc]) != -1.0:
                        out.append((nr, nc))
        return out

    def _next_position(self, xi, xp, neighbours):
        """Greedy pick: argmax( xj_value + c · (1 - d/√2) )."""
        best       = neighbours[0]
        best_score = -float('inf')
        inv_sqrt2  = 1.0 / math.sqrt(2.0)
        for nb in neighbours:
            d        = math.hypot(xi[0] - nb[0], xi[1] - nb[1])
            cost_val = 1.0 - d * inv_sqrt2
            score    = float(self._grid[nb[0], nb[1]]) + self.c * cost_val
            if score > best_score:
                best_score = score
                best       = nb
        return best