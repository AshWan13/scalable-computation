#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gbnnh.py
Standalone GBNN+H: Complete Area-Coverage Path Planning for Surface Cleaning
with a Mobile Dual-Arm (MDA) Robot.

Filename note
-------------
The module file is named ``gbnnh.py`` (lowercase, no '+') so it imports
cleanly via::

    from gbnnh import GBNN_H, RoIFrame, AccessPoint

— matching how ``Configurer.py`` and ``Interstar.py`` are loaded by
``demo.py``.  Class names, docstrings, and paper citations retain the
original "GBNN+H" notation.

Implements the GBNN+H algorithm (GBNN with Heuristics) as described in:

    Wan et al., "Complete area-coverage path planner for surface cleaning
    in hospital settings using mobile dual-arm robot and GBNN with heuristics",
    Complex & Intelligent Systems 10.5 (2024), pp. 6767-6785.

Key idea (vs. base GBNN)
------------------------
Waypoint selection (Eqn 6) is biased by a dynamic heuristic term:

    score(neighbour j) = max( N_j+ + c * cost_value + rp * H , 0 )

where
    N_j+        positive neural activity at j (standard GBNN)
    cost_value  static direction/distance preference
                  cost = (1 - d(xi,j)/sqrt(2)) / pi
    H           target-pull term: d(xi,G) - d(j,G)
    G           target point (nearest unvisited cell, or COG of unvisited)
    rp          per-EE dynamic coefficient:
                  starts at rp_start; increments by rp_increment each step
                  the EE is OFF-dirt; resets to 0 when EE lands on a dirt cell.
                  This drives the EE out of deadlocks and over gaps.

Grid encoding
-------------
     1.0 = unvisited (dirty / on-surface)
    -1.0 = obstacle / off-surface (still traversable, counts as flight)
    other = visited (decay mode: cell holds its neural-sum value and still
            propagates influence to neighbours via w_ij)

Robot model
-----------
Single Mobile Dual-Arm (MDA) robot with two end-effectors EE_L, EE_R.
Each EE gets its own workspace (partition of the grid, optionally overlapping).
Both EEs step simultaneously every tick.
EE mode label (for visualisation only):
    I = Impedance (EE on surface, cell != -1)
    P = Position  (EE on obstacle/off-surface, cell == -1)  --> flight step

Usage
-----
    from gbnnh import GBNN_H, make_grid

    grid    = make_grid(12, 30, obstacle_chance=0.15, seed=0)  # any (H, V)
    planner = GBNN_H(grid, n_ee=2, visualize=False)
    paths, stats = planner.run()

    # paths: dict  {ee_id: [(x, y), ...]}
    # stats: {'steps': int, 'flight': {0: int, 1: int},
    #         'coverage': float, 'wall_time': float, 'stop_reason': str}

Grid shape
----------
The class accepts any 2-D array of shape (H, V) -- H and V do NOT need to
be equal, and there is no upper or lower bound enforced.  Partition defaults
(split_at, initial_positions) are derived from the actual grid dimensions.
"""

import math
import time
import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Matplotlib is intentionally NOT imported at module load.  Both the per-step
# GBNN_H rendering (`_viz`) and the floor view (`_viz_floor`) import it lazily
# so that pygame / headless / ROS2 consumers never pay the matplotlib import
# cost just to use the planner classes.


# ============================================================================
#  GBNN_H
# ============================================================================

class GBNN_H:
    """
    Standalone GBNN+H dual-EE complete-coverage path planner.

    Parameters
    ----------
    grid : np.ndarray, shape (H, V)
        2-D surface map of ANY shape (H != V is fine).
        1.0 = unvisited free, -1.0 = obstacle/off-surface.
    n_ee : int
        Number of end-effectors (1 or 2).  Default 2.
    split_axis : 'y' or 'x'
        Partition axis for workspace assignment.  'y' = vertical split
        (left/right halves), 'x' = horizontal split (top/bottom halves).
        Ignored when n_ee == 1.
    split_at : int or None
        Column (or row) at which to split.  None -> middle of the grid.
    overlap : int
        Overlap band (in cells) shared by both EEs at the split boundary.
    E : float
        GBNN bias magnitude for unvisited / obstacle cells (Eqn 5).
    r : float
        Receptive-field radius for w_ij (Eqn 4).
    alpha : float
        Decay constant in w_ij (Eqn 4).  Called ``a`` / ``q`` in the paper.
    b : float
        Slope of activity function f(x) on [0,1) (Eqn 2).
    c : float
        Static cost-term coefficient on the O term in waypoint selection
        (Eqn 12).  Default 1.0 so the code literally matches
        wp_{i+1} = max(N_k + O + H_val) from the paper.  Setting c=0.01
        (the reference-simulation value) down-weights O so it does not
        dominate early in the run.
    rp_start : float
        Initial value of the per-EE heuristic coefficient C (Eqn 11).
        Default 0.0 so the heuristic is dormant until the EE's first
        off-dirt step, consistent with the paper's "activated over
        iterations where the planner is successively unable to obtain a
        selected wp_{i+1}" description on p. 13.  Set to 1.0 to match the
        reference simulation (heuristic active from step 0).
    rp_increment : float
        Amount rp grows by each step the EE is OFF dirt.
    rp_reset_on_dirt : bool
        If True, rp resets to 0 whenever the EE steps onto an unvisited cell.
    c_decay_step : int or None
        At this step count, c is set to 0 (long-tail fallback).  None disables.
    h_target : 'nearest' or 'cog'
        How to compute target G per EE.
    visited_mode : 'decay'
        Visited cells take their neural-sum value (paper-faithful).
        ('zero' reserved for future ablation; not used here.)
    obstacle_mode : 'traversable' or 'blocked'
        Whether EEs may step onto -1 cells (traversable = paper-faithful).
    initial_positions : list of (row, col) tuples or None
        Starting cells per EE.  None -> defaults to [(1, 0), (1, V-1)] for
        n_ee==2, or [(1, 0)] for n_ee==1.
    step_cap : int or None
        Hard step budget.  None -> no cap; coverage runs to completion.
    visualize : bool
        Render the grid each tick.  Set False for batch / benchmarking.
    viz_interval : int
        Render only every N ticks when visualize=True.
    """

    # GBNN cell-colour palette
    COLOR_OBSTACLE  = '#0A090A'   # black
    COLOR_UNVISITED = '#C1C1C1'   # light grey
    COLOR_VISITED   = '#5F7472'   # slate grey
    # Per-EE accent colours
    EE_COLORS = ['blue', 'green', 'red', 'cyan', 'magenta',
                 'orange', 'purple', 'brown', 'pink', 'olive']

    _DIRS8 = [(0, -1), (0, 1), (-1, 0), (1, 0),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(self, grid, n_ee=2,
                 split_axis='y', split_at=None, overlap=0,
                 E=100.0, r=2.0, alpha=2.0, b=0.7, c=1.0,
                 rp_start=0.0, rp_increment=0.1, rp_reset_on_dirt=True,
                 c_decay_step=8100,
                 h_target='nearest', visited_mode='decay',
                 obstacle_mode='traversable',
                 initial_positions=None,
                 step_cap=None, visualize=True, viz_interval=1):

        self.grid = np.array(grid, dtype=float)
        if self.grid.ndim == 3 and self.grid.shape[2] == 1:
            self.grid = self.grid[:, :, 0]          # tolerate (H,V,1) input
        if self.grid.ndim != 2:
            raise ValueError(f"grid must be 2-D; got shape {self.grid.shape}")

        self.H, self.V = self.grid.shape
        if n_ee not in (1, 2):
            raise ValueError("n_ee must be 1 or 2")
        self.n_ee = n_ee

        # Partition
        self.split_axis = split_axis
        self.overlap    = overlap
        if split_at is None:
            self.split_at = (self.V // 2) if split_axis == 'y' else (self.H // 2)
        else:
            self.split_at = split_at

        # GBNN core
        self.E     = E
        self.r     = r
        self.alpha = alpha
        self.b     = b

        # Selection / heuristic params
        self.c_init           = c
        self.rp_start         = rp_start
        self.rp_increment     = rp_increment
        self.rp_reset_on_dirt = rp_reset_on_dirt
        self.c_decay_step     = c_decay_step
        self.h_target         = h_target
        self.visited_mode     = visited_mode
        self.obstacle_mode    = obstacle_mode

        # Run control
        self.step_cap     = step_cap
        self.visualize    = visualize
        self.viz_interval = max(1, int(viz_interval))

        # Initial EE positions
        if initial_positions is None:
            if n_ee == 2:
                initial_positions = [(1, 0), (1, self.V - 1)]
            else:
                initial_positions = [(1, 0)]
        if len(initial_positions) != n_ee:
            raise ValueError("initial_positions length must equal n_ee")
        self.ee_pos = [list(p) for p in initial_positions]

        # Per-EE dynamic state
        self.rp         = [rp_start] * n_ee
        self.flight     = [0] * n_ee
        self.path_plans = {i: [] for i in range(n_ee)}

        # Runtime
        self._c_now = c
        self._step  = 0

    # ==================================================================
    #  SECTION 1 - GBNN CORE  (Eqns 2-5)
    # ==================================================================

    def _w(self, xi, xj):
        """Connection weight between neurons xi and xj (Eqn 4)."""
        d = math.sqrt((xi[0] - xj[0]) ** 2 + (xi[1] - xj[1]) ** 2)
        return math.exp(-self.alpha * d * d) if 0.0 < d < self.r else 0.0

    def _f(self, x):
        """Activity function f(x) (Eqn 2)."""
        if x < 0.0:  return -1.0
        if x >= 1.0: return 1.0
        return self.b * x

    def _cell_ee(self, i, j):
        """Which EE owns cell (i, j)?  Returns ee_id, or None if no owner.

        For overlap > 0, both EEs own the overlap band; we return the lower
        id (consistent with reference behaviour where EE-0 processes first).
        """
        if self.n_ee == 1:
            return 0
        if self.split_axis == 'y':
            if j < self.split_at + self.overlap:
                return 0
            return 1
        else:   # 'x'
            if i < self.split_at + self.overlap:
                return 0
            return 1

    def _neuron(self, i, j):
        """
        Recompute activity at (i,j) per Eqns 1, 3, 4.

        The neighbour sum is taken over ALL 8-connected cells inside the
        grid -- it is NOT partitioned by EE workspace.  This lets positive
        activity propagate across the midline between EE_L and EE_R, which
        matches Algorithm 1 (the paper partitions only ``GetNeighbors`` at
        waypoint-selection time, not the neural update).
        """
        v  = self.grid[i, j]
        s  = 0.0
        for dx, dy in self._DIRS8:
            ni, nj = i + dx, j + dy
            if not (0 <= ni < self.H and 0 <= nj < self.V):
                continue
            s += self._w((i, j), (ni, nj)) * max(self.grid[ni, nj], 0.0)
        Ii = self.E if v == 1.0 else (-self.E if v == -1.0 else 0.0)
        return self._f(s + Ii)

    def _update_grid(self):
        """
        Full in-place neural activity pass over the whole grid.

        In-place writes mean later cells see already-updated neighbours --
        this matches the reference ``update_grid`` exactly.  Activity
        propagates globally; the EE-workspace partition is applied only in
        ``_get_neighbours`` (selection), consistent with Algorithm 1.
        """
        for i in range(self.H):
            for j in range(self.V):
                self.grid[i, j] = self._neuron(i, j)

    def _update_visited(self, xi):
        """
        Decay-mode visited write: the cell takes the raw neighbour
        neural-sum value (no bias, no f-clamp), so it keeps propagating to
        neighbours.  Mirrors ``update_xi_to_covered`` in the reference.

        The neighbour sum is unpartitioned -- consistent with _neuron.
        """
        s = 0.0
        for dx, dy in self._DIRS8:
            ni, nj = xi[0] + dx, xi[1] + dy
            if not (0 <= ni < self.H and 0 <= nj < self.V):
                continue
            s += self._w(xi, (ni, nj)) * max(self.grid[ni, nj], 0.0)
        self.grid[xi[0], xi[1]] = s

    # ==================================================================
    #  SECTION 2 - WORKSPACE PARTITION & NEIGHBOURS
    # ==================================================================

    def _ee_bounds(self, ee_id):
        """Axis-aligned bounds owned by ee_id: (r_lo, r_hi, c_lo, c_hi)."""
        if self.n_ee == 1:
            return 0, self.H, 0, self.V
        if self.split_axis == 'y':
            if ee_id == 0:
                return 0, self.H, 0, self.split_at + self.overlap
            return 0, self.H, self.split_at - self.overlap, self.V
        else:   # 'x'
            if ee_id == 0:
                return 0, self.split_at + self.overlap, 0, self.V
            return self.split_at - self.overlap, self.H, 0, self.V

    def _in_ee_region(self, i, j, ee_id):
        """Is cell (i,j) inside EE ee_id's workspace?"""
        if ee_id is None:
            return False
        r_lo, r_hi, c_lo, c_hi = self._ee_bounds(ee_id)
        return r_lo <= i < r_hi and c_lo <= j < c_hi

    def _get_neighbours(self, xi, ee_id):
        """
        8-connected neighbours of xi that lie inside EE's workspace.
        When obstacle_mode == 'blocked', -1 cells are excluded.
        Always excludes the OTHER EE's current cell (collision guard).
        """
        cx, cy = xi[0], xi[1]
        other  = [self.ee_pos[k] for k in range(self.n_ee)
                  if self.ee_pos[k] != list(xi)]
        nbs = []
        for dx, dy in self._DIRS8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self.H and 0 <= ny < self.V):
                continue
            if not self._in_ee_region(nx, ny, ee_id):
                continue
            if self.obstacle_mode == 'blocked' and self.grid[nx, ny] == -1.0:
                continue
            if [nx, ny] in other:
                continue
            nbs.append([nx, ny])
        if not nbs:
            nbs = [[cx, cy]]              # fallback: stay
        return nbs

    # ==================================================================
    #  SECTION 3 - HEURISTIC  (target G + H term)
    # ==================================================================

    def _find_target(self, ee_id):
        """
        Compute target point G for EE ee_id.
            'nearest' : closest unvisited cell (grid == 1) in EE's region
            'cog'     : centre of gravity of all unvisited cells
        Returns (gx, gy) or None if EE's region is fully covered.
        """
        r_lo, r_hi, c_lo, c_hi = self._ee_bounds(ee_id)
        xi = self.ee_pos[ee_id]

        if self.h_target == 'nearest':
            best_d, best = float('inf'), None
            for i in range(r_lo, r_hi):
                for j in range(c_lo, c_hi):
                    if self.grid[i, j] == 1.0:
                        d = (xi[0] - i) ** 2 + (xi[1] - j) ** 2
                        if d < best_d:
                            best_d, best = d, (i, j)
            return best

        if self.h_target == 'cog':
            sx = sy = n = 0
            for i in range(r_lo, r_hi):
                for j in range(c_lo, c_hi):
                    if self.grid[i, j] == 1.0:
                        sx += i; sy += j; n += 1
            if n == 0:
                return None
            return (sx / n, sy / n)

        raise ValueError(f"unknown h_target: {self.h_target}")

    @staticmethod
    def _euclid(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # ==================================================================
    #  SECTION 4 - WAYPOINT SELECTION  (Eqn 6)
    # ==================================================================

    def _cost_value(self, xi, xj):
        """Static direction/distance preference, as in reference."""
        d = self._euclid(xi, xj)
        return (1.0 - d / math.sqrt(2.0)) / math.pi

    def _next_wp(self, ee_id):
        """
        Select next waypoint for EE ee_id per Eqn 12 of the paper:

            wp_{i+1} = argmax_j ( N_j+ + c*O(xi,j) + C * H_k(xi,j,G) )

        where C is the per-EE heuristic coefficient (self.rp) and c is the
        scalar weight on O (default 1.0 -- paper-literal; the reference
        simulation used 0.01).  No inner clamp: the raw sum is maximised.

        After selection, update C per the paper's dynamics (p. 13):
          - C resets to 0 when the selected cell has neural activity == 1
            (i.e. the EE landed on dirt).  NOTE: the paper sentence on
            p. 13 literally reads "...where the neural activity is not
            equivalent to 1, the value of C will be reset..."  That
            wording is inverted from the intended algorithmic behaviour
            (resetting on every off-dirt step would nullify the heuristic).
            The code implements the intended reading: reset when
            activity IS 1.
          - Otherwise C += rp_increment (heuristic grows while off dirt).
        """
        xi  = self.ee_pos[ee_id]
        nbs = self._get_neighbours(xi, ee_id)
        G   = self._find_target(ee_id)
        rp  = self.rp[ee_id]
        c   = self._c_now

        scores = []
        for nb in nbs:
            xj_val = self.grid[nb[0], nb[1]]
            cost   = self._cost_value(xi, nb)
            if G is None:
                H = 0.0
            else:
                H = self._euclid(xi, G) - self._euclid(nb, G)
            # Paper Eqn 12: raw sum, no inner clamp.
            scores.append(xj_val + c * cost + rp * H)

        best = nbs[scores.index(max(scores))]

        # --- C dynamics (see docstring for the paper-phrasing footnote) ---
        landed_on_dirt = (self.grid[best[0], best[1]] == 1.0)
        if landed_on_dirt and self.rp_reset_on_dirt:
            self.rp[ee_id] = 0.0
        else:
            self.rp[ee_id] += self.rp_increment

        # --- flight accounting: landing on obstacle/off-surface ---
        if self.grid[best[0], best[1]] == -1.0:
            self.flight[ee_id] += 1

        return best

    # ==================================================================
    #  SECTION 5 - VISUALISATION
    # ==================================================================

    def _viz(self, label=""):
        """
        Per-iteration render.  Reuses a single persistent figure window
        (named "GBNN+H") across iterations: each call clears the figure
        and redraws on the same window, advancing every ~0.1 s without
        flicker.

        Palette:
            grid[i,j] == -1.0  -> obstacle  (black  #0A090A)
            grid[i,j] ==  1.0  -> unvisited (light  #C1C1C1)
            else               -> visited   (slate  #5F7472)
                                 (decay-mode visited cells keep their
                                  neural-sum value and still propagate)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.ion()
        fig = plt.figure(num='GBNN+H', figsize=(10, 10))
        fig.clf()
        ax = fig.add_subplot(111)

        # --- Cell background ---
        for i in range(self.H):
            for j in range(self.V):
                v = self.grid[i, j]
                if v == -1.0:
                    c = self.COLOR_OBSTACLE
                elif v == 1.0:
                    c = self.COLOR_UNVISITED
                else:
                    c = self.COLOR_VISITED
                ax.add_patch(patches.Rectangle(
                    (j - .5, i - .5), 1, 1, color=c))

        # --- EE partition line (thin dotted red) ---
        if self.n_ee == 2:
            if self.split_axis == 'y':
                xline = self.split_at - 0.5
                ax.plot([xline, xline], [-0.5, self.H - 0.5],
                        linestyle=':', color='red', linewidth=0.8)
            else:   # 'x'
                yline = self.split_at - 0.5
                ax.plot([-0.5, self.V - 0.5], [yline, yline],
                        linestyle=':', color='red', linewidth=0.8)

        # --- EE markers (square patch + centre dot), with I/P mode label ---
        mode_labels = []
        for k, (ex, ey) in enumerate(self.ee_pos):
            on_surface = (self.grid[ex, ey] != -1.0)
            mode       = 'I' if on_surface else 'P'
            mode_labels.append(f'EE{k}={mode}')
            col = self.EE_COLORS[k % len(self.EE_COLORS)]
            ax.add_patch(patches.Rectangle(
                (ey - .5, ex - .5), 1, 1, color=col, alpha=0.85))
            ax.scatter(ey, ex, c='white', s=350, zorder=5)
            ax.scatter(ey, ex, c=col, s=350, marker='.', zorder=6)

        ax.set_xlim(-.5, self.V - .5)
        ax.set_ylim(-.5, self.H - .5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.title(f'GBNN+H  |  {label}  |  Step {self._step}  |  '
                  f'{"  ".join(mode_labels)}',
                  fontsize=13)
        plt.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(0.1)

    # ==================================================================
    #  SECTION 6 - RUN LOOP
    # ==================================================================

    def _complete(self):
        """True iff no unvisited (==1.0) cell remains anywhere."""
        return not np.any(self.grid == 1.0)

    def _initial_marking(self):
        """Mark each EE's starting cell as visited (decay-mode write)."""
        for k in range(self.n_ee):
            ex, ey = self.ee_pos[k]
            if self.grid[ex, ey] != -1.0:
                self._update_visited((ex, ey))
            self.path_plans[k].append((ex, ey))

    # ------------------------------------------------------------------
    #  Tick-driven API  (reset + step + final_result)
    # ------------------------------------------------------------------
    #
    #  These methods expose the run loop one iteration at a time so
    #  external drivers (pygame Mode 5, ROS2 action servers, headless
    #  benchmarks) can advance the planner on their own clock without
    #  blocking inside `run()`.  The legacy `run()` below is preserved
    #  for backward compatibility and for the standalone `__main__` demo
    #  — it now delegates to reset()/step()/final_result().
    #
    #  Lifecycle:
    #     planner = GBNN_H(grid, ...)
    #     planner.reset()                  # first tick is now legal
    #     while planner.step(): pass       # advance until done
    #     paths, stats = planner.final_result()
    #
    #  After reset() the planner exposes live state via render_state()
    #  (see below); this is the contract pygame consumes.

    def reset(self):
        """Prepare the planner for tick-driven execution.

        Performs the prologue work formerly inlined at the top of
        `run()`: initial marking, first global neural-activity pass,
        coverage-metric bookkeeping.  After this call, `step()` is
        safe to invoke; before it, `step()` raises.
        """
        self._t0 = time.time()
        self._total_dirty = int(np.sum(self.grid == 1.0))
        # Starting cells on surface count toward coverage; preserved
        # only for parity with the original run() narrative — coverage
        # is recomputed from the live grid in final_result().
        self._start_covered = sum(
            1 for (ex, ey) in self.ee_pos if self.grid[ex, ey] == 1.0
        )

        self._initial_marking()
        self._update_grid()
        self._step = 0
        self._done = self._complete()
        self._stop_reason = "complete" if self._done else None
        self._started = True

        if self.visualize:
            self._viz(label="init")

    def step(self) -> bool:
        """Advance one iteration of the GBNN+H loop.

        Returns
        -------
        running : bool
            True if more work remains (caller should call step() again).
            False if the planner has terminated (coverage complete OR
            step_cap reached).  Once False, subsequent step() calls are
            no-ops that keep returning False.

        Raises
        ------
        RuntimeError
            If called before `reset()`.
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "GBNN_H.step() called before reset(); call reset() first."
            )
        if self._done:
            return False

        # c-decay fallback (paper: step 8100 -> c=0)
        if (self.c_decay_step is not None and
                self._step >= self.c_decay_step):
            self._c_now = 0.0

        # Step all EEs simultaneously
        new_positions = [self._next_wp(k) for k in range(self.n_ee)]

        # Commit moves + visited writes
        for k, new in enumerate(new_positions):
            self.ee_pos[k] = new
            if self.grid[new[0], new[1]] != -1.0:
                self._update_visited(tuple(new))
            self.path_plans[k].append(tuple(new))

        # Neural activity refresh
        self._update_grid()
        self._step += 1

        if self.visualize and (self._step % self.viz_interval == 0):
            self._viz(label="running")

        # Termination checks
        if self._complete():
            self._done = True
            self._stop_reason = "complete"
        elif self.step_cap is not None and self._step >= self.step_cap:
            self._done = True
            self._stop_reason = "step_cap"
            print(f"[GBNN+H] step_cap={self.step_cap} reached.")

        return not self._done

    def final_result(self):
        """Build the (paths, stats) return tuple after termination.

        Safe to call multiple times.  Idempotent.  Includes the same
        keys produced by the legacy `run()` finale.
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "GBNN_H.final_result() called before reset()."
            )
        if self.visualize and self._done:
            # Render terminal state once.  Guarded so headless (visualize
            # off) consumers and repeated final_result() calls don't
            # spam matplotlib.
            if not getattr(self, "_finalised_viz", False):
                self._viz(label="complete")
                self._finalised_viz = True

        wall = time.time() - getattr(self, "_t0", time.time())
        covered = self._total_dirty - int(np.sum(self.grid == 1.0))
        coverage = covered / max(self._total_dirty, 1)

        stats = {
            'steps': self._step,
            'flight': {k: self.flight[k] for k in range(self.n_ee)},
            'coverage': coverage,
            'wall_time': wall,
            'stop_reason': self._stop_reason or "running",
        }
        return self.path_plans, stats

    def is_done(self) -> bool:
        """True once step() has terminated (coverage or step_cap)."""
        return bool(getattr(self, "_done", False))

    def render_state(self) -> Dict[str, Any]:
        """Snapshot of live planner state for external renderers.

        Returns a fresh dict per call.  The grid is a defensive copy so
        the caller can hold on to it across ticks without aliasing.
        Designed for pygame's surface-view subpanel: enough info to
        render cells, EE markers, and the partition line.
        """
        return {
            'shape':       (self.H, self.V),
            'grid':        self.grid.copy(),
            'ee_pos':      [tuple(p) for p in self.ee_pos],
            'flight':      list(self.flight),
            'paths':       {k: list(self.path_plans[k])
                            for k in range(self.n_ee)},
            'step':        self._step,
            'done':        self.is_done(),
            'stop_reason': getattr(self, "_stop_reason", None),
            'split_axis':  self.split_axis,
            'split_at':    self.split_at,
            'overlap':     self.overlap,
            'n_ee':        self.n_ee,
        }

    # ------------------------------------------------------------------
    #  Legacy synchronous run()
    # ------------------------------------------------------------------

    def run(self):
        """
        Main loop (legacy entry point — preserved for backward compat
        and standalone demos).  Now delegates to reset/step/final_result.

        Returns
        -------
        paths : dict {ee_id: [(x, y), ...]}
        stats : dict with keys 'steps', 'flight', 'coverage', 'wall_time',
                'stop_reason'
        """
        print(f"[GBNN+H] Grid {self.H}x{self.V} | n_ee={self.n_ee} | "
              f"split_axis={self.split_axis} split_at={self.split_at} "
              f"overlap={self.overlap} | h_target={self.h_target}")

        self.reset()
        while self.step():
            pass
        paths, stats = self.final_result()

        print(f"[GBNN+H] Done. steps={stats['steps']}  "
              f"flight={self.flight}  coverage={stats['coverage']:.3f}  "
              f"wall={stats['wall_time']:.2f}s")
        return paths, stats


# ============================================================================
#  ACCESS POINT + MOBILE PLATFORM
# ============================================================================

# ============================================================================
#  RoIFrame  —  surface-local (i, j) ↔ world (x, y) mapping for an AP
# ============================================================================

@dataclass
class RoIFrame:
    """
    Maps surface-local cell coordinates to world coordinates for one AP.

    The surface is treated as a 2-D rectangular patch standing in front of
    the MDA base.  At the AP, the base sits at ``anchor_xy`` looking along
    yaw ``yaw_rad`` (this is the "surface-normal direction" — i.e. the
    direction the arms reach to touch the surface).

    Surface frame convention
    ------------------------
    * The grid has shape (H, V) — H rows (i), V columns (j).
    * Surface origin (i=0, j=0) sits at the **far-left** corner of the
      surface as seen from the MDA base, ``reach_offset`` ahead of the
      anchor along the yaw direction.
    * Increasing j moves to the right along the surface (perpendicular to
      yaw, with right defined by a -90° rotation of yaw).
    * Increasing i moves "down" the surface — perpendicular to the row
      axis, in the same plane as the surface.  For a vertical surface
      (e.g. wall), this is the world's -z direction; for a horizontal
      surface (e.g. tabletop), it is back into the surface away from the
      base (yaw direction).
    * For Phase E pygame integration we use the projected-floor mapping:
      i runs along the yaw direction (away from base), j runs along the
      perpendicular (right of base).  That keeps the surface-view panel
      and the floor view consistent for floor-projected RoIs.

    Attributes
    ----------
    anchor_xy : (x, y)
        AP base position in world coords.
    yaw_rad : float
        Surface-normal direction in world frame (radians).
    cell_size : float
        World units per cell (square cells).
    grid_shape : (H, V)
        Surface grid dimensions.
    reach_offset : float
        Distance from base to surface-origin (cell (0, 0)) along yaw.
        Defaults to 0.0 — set positive when the surface is in front of
        the base (typical hospital-bed cleaning).
    """
    anchor_xy:    Tuple[float, float]
    yaw_rad:      float
    cell_size:    float
    grid_shape:   Tuple[int, int]
    reach_offset: float = 0.0

    def cell_to_world(self, i: float, j: float) -> Tuple[float, float]:
        """Convert surface-local (i, j) to world (x, y).

        Accepts floats so cell centres (i+0.5, j+0.5) and arbitrary
        sub-cell positions both map cleanly.
        """
        ax, ay  = self.anchor_xy
        cs      = self.cell_size
        H, V    = self.grid_shape
        # Forward axis (yaw direction) carries i, plus reach_offset.
        # Right axis (yaw - 90°) carries j with j=0 at the LEFT edge of
        # the surface, hence the -V/2 centring.
        forward = self.reach_offset + (i + 0.5) * cs
        right   = (j + 0.5 - V / 2.0) * cs
        cyaw, syaw = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        # World = anchor + forward * yaw_dir + right * (yaw - 90°)
        x = ax + forward * cyaw + right * syaw
        y = ay + forward * syaw - right * cyaw
        return (x, y)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Inverse of cell_to_world; returns the integer (i, j) cell.

        Out-of-grid coordinates are clamped to the nearest valid cell.
        Uses floor semantics so the cell whose centre is at
        ``cell_to_world(i, j)`` round-trips back to (i, j).
        """
        ax, ay = self.anchor_xy
        cs     = self.cell_size
        H, V   = self.grid_shape
        cyaw, syaw = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        dx, dy = x - ax, y - ay
        forward = dx * cyaw + dy * syaw
        right   = dx * syaw - dy * cyaw
        i = int(math.floor((forward - self.reach_offset) / cs))
        j = int(math.floor(right / cs + V / 2.0))
        i = max(0, min(H - 1, i))
        j = max(0, min(V - 1, j))
        return (i, j)

    def grid_polygon(self) -> List[Tuple[float, float]]:
        """Return the four corner points (world frame) of the surface
        rectangle, ordered as: near-left, near-right, far-right, far-left.

        Useful for rendering the surface footprint on the floor view.
        """
        H, V = self.grid_shape
        # Use cell-edge corners: i in {0, H}, j in {0, V} mapped via
        # cell_to_world(i-0.5, j-0.5) so the rectangle covers the whole
        # grid extent rather than just cell centres.
        corners_ij = [(-0.5, -0.5),
                      (-0.5, V - 0.5),
                      (H - 0.5, V - 0.5),
                      (H - 0.5, -0.5)]
        return [self.cell_to_world(i, j) for (i, j) in corners_ij]


# ============================================================================
#  ACCESS POINT + MOBILE PLATFORM
# ============================================================================

@dataclass
class AccessPoint:
    """
    One fiducial-marked waypoint for the MDA (Section 2.3 / Fig 3(b)).

    Attributes
    ----------
    pose : (x, y, theta)
        2-D pose of the mobile base when aligned to this RoI.  theta in rad.
    roi_grid : np.ndarray
        2-D GBNN+H input grid for the cleaning surface reachable from `pose`.
        Encoding: 1.0 unvisited, -1.0 obstacle/off-surface.
    label : str
        Optional human-readable tag (e.g. "bed_head", "side_table").
    gbnn_kwargs : dict
        Per-AP overrides for GBNN_H constructor (e.g. initial_positions,
        n_ee, split_axis).  Merged over the platform-level defaults.
    """
    pose: Tuple[float, float, float]
    roi_grid: np.ndarray
    label: str = ""
    gbnn_kwargs: dict = field(default_factory=dict)


@dataclass
class FloorMap:
    """
    Top-down floor/environment description for MobilePlatform visualisation.

    Attributes
    ----------
    bounds : (xmin, ymin, xmax, ymax)
        Rectangular extent of the floor in world units.
    tables : list of (x, y, w, h)
        Axis-aligned rectangular obstacles (tables, beds, etc.),
        where (x, y) is the bottom-left corner and (w, h) are extents.
    walls : list of ((x1, y1), (x2, y2))
        Optional line segments representing walls.
    grid_step : float
        Gridline spacing for the floor plot (set 0 to disable).
    """
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)
    tables: List[Tuple[float, float, float, float]] = field(default_factory=list)
    walls:  List[Tuple[Tuple[float, float], Tuple[float, float]]] = \
        field(default_factory=list)
    grid_step: float = 1.0


class MobilePlatform:
    """
    MDA mobile-base sequencer for multi-RoI cleaning tasks.

    Implements the access-point loop from Section 2.3 / Fig 3(b):

        for each access point ap in access_points:
            1. move_to(ap.pose)                   # holonomic drive
            2. planner = GBNN_H(ap.roi_grid, ...) # build per-AP planner
            3. paths, stats = planner.run()       # cleaning pass
            4. wait for complete flag             # coverage >= threshold
        done when all APs report complete

    The base is modelled as holonomic: translation and rotation proceed
    concurrently, consistent with the paper's MDA ("sideways movement,
    access tight spaces", Section 2.2).

    Parameters
    ----------
    access_points : list of AccessPoint
        Ordered sequence of waypoints to attend.
    initial_pose : (x, y, theta)
        Starting pose of the base in world frame.
    linear_speed : float
        Base translation speed (units / sec).
    angular_speed : float
        Base rotation speed (rad / sec).
    dt : float
        Trajectory sampling interval (sec) for the move_to log.
    completion_threshold : float
        Coverage fraction at which an AP is considered complete.
        Default 1.0 (full coverage).
    max_retries : int
        If an AP's GBNN+H pass does not reach completion_threshold,
        retry up to max_retries times before giving up and advancing.
    gbnn_kwargs : dict
        Default keyword arguments passed to GBNN_H at each AP
        (overridden by per-AP AccessPoint.gbnn_kwargs).
    verbose : bool
        Print per-AP progress.
    floor_map : FloorMap or None
        Top-down environment (tables, bounds).  Used only for visualisation.
        If None and visualize=True, bounds are inferred from AP positions.
    visualize : bool
        If True, render a floor-view frame per motion sample and hand off
        to GBNN_H's per-step viz during each AP's cleaning phase.
    motion_frames : int
        Number of floor-view frames to render per move_to call.  0 disables
        motion animation (still renders an 'arrival' frame at each AP).
    show_gbnn : bool
        Whether GBNN+H runs with visualize=True at each AP.  When True,
        overrides 'visualize' in gbnn_kwargs.
    nav_resolution : float
        Cell size of the A* occupancy grid used for obstacle avoidance
        during move_to.  Smaller = smoother paths, higher cost.
    robot_radius : float
        Robot footprint radius; tables/walls are inflated by this amount
        so the planned path clears them.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(self,
                 access_points: List[AccessPoint],
                 initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 linear_speed: float = 0.5,
                 angular_speed: float = 0.5,
                 dt: float = 0.1,
                 completion_threshold: float = 1.0,
                 max_retries: int = 0,
                 gbnn_kwargs: Optional[dict] = None,
                 verbose: bool = True,
                 floor_map: Optional[FloorMap] = None,
                 visualize: bool = False,
                 motion_frames: int = 8,
                 show_gbnn: bool = True,
                 nav_resolution: float = 0.1,
                 robot_radius: float = 0.35):

        if not access_points:
            raise ValueError("access_points must be non-empty")
        self.access_points = list(access_points)

        self.pose               = tuple(initial_pose)
        self.linear_speed       = float(linear_speed)
        self.angular_speed      = float(angular_speed)
        self.dt                 = float(dt)
        self.completion_threshold = float(completion_threshold)
        self.max_retries        = int(max_retries)
        self.gbnn_kwargs        = dict(gbnn_kwargs) if gbnn_kwargs else {}
        self.verbose            = bool(verbose)

        # Visualisation
        self.floor_map          = floor_map
        self.visualize          = bool(visualize)
        self.motion_frames      = int(motion_frames)
        self.show_gbnn          = bool(show_gbnn)

        # Navigation (A* obstacle avoidance)
        self.nav_resolution     = float(nav_resolution)
        self.robot_radius       = float(robot_radius)
        self._occ_cache         = None          # cached occupancy grid
        self._last_planned_path = []            # last planned polyline

        # Logs
        self.trajectory = [tuple(initial_pose)]        # base poses over time
        self.ap_log: List[dict] = []                    # per-AP results

    # ==================================================================
    #  SECTION 1 - OBSTACLE-AVOIDING PATH PLANNING
    # ==================================================================

    @staticmethod
    def _wrap_angle(a):
        """Wrap angle to (-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    def _build_occupancy(self):
        """
        Build (or fetch cached) inflated occupancy grid from floor_map.

        Returns (occ, origin_xy, resolution) where occ[r, c] == 1 marks
        cells too close to a table/wall for the robot centre to occupy.
        Returns None if there is no floor_map.
        """
        if self.floor_map is None:
            return None
        if self._occ_cache is not None:
            return self._occ_cache

        xmin, ymin, xmax, ymax = self.floor_map.bounds
        res  = self.nav_resolution
        infl = self.robot_radius
        W = max(1, int(math.ceil((xmax - xmin) / res)))
        H = max(1, int(math.ceil((ymax - ymin) / res)))
        occ = np.zeros((H, W), dtype=np.uint8)

        # Inflate each table rectangle by robot_radius, rasterise
        for (tx, ty, tw, th_) in self.floor_map.tables:
            x0 = max(0, int(math.floor((tx - infl - xmin) / res)))
            x1 = min(W, int(math.ceil((tx + tw + infl - xmin) / res)))
            y0 = max(0, int(math.floor((ty - infl - ymin) / res)))
            y1 = min(H, int(math.ceil((ty + th_ + infl - ymin) / res)))
            if x1 > x0 and y1 > y0:
                occ[y0:y1, x0:x1] = 1

        # Walls: mark any cell whose centre lies within robot_radius
        # of the line segment.
        if self.floor_map.walls:
            for (p1, p2) in self.floor_map.walls:
                self._rasterise_wall(occ, p1, p2, (xmin, ymin), res, infl)

        self._occ_cache = (occ, (xmin, ymin), res)
        return self._occ_cache

    @staticmethod
    def _rasterise_wall(occ, p1, p2, origin, res, infl):
        """Mark cells within `infl` of segment p1-p2 as occupied."""
        ox, oy = origin
        H, W = occ.shape
        # Iterate a rectangular region around the segment for efficiency
        xmin = min(p1[0], p2[0]) - infl
        xmax = max(p1[0], p2[0]) + infl
        ymin = min(p1[1], p2[1]) - infl
        ymax = max(p1[1], p2[1]) + infl
        c0 = max(0, int((xmin - ox) / res))
        c1 = min(W, int(math.ceil((xmax - ox) / res)))
        r0 = max(0, int((ymin - oy) / res))
        r1 = min(H, int(math.ceil((ymax - oy) / res)))
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L2 = dx * dx + dy * dy
        for r in range(r0, r1):
            for c in range(c0, c1):
                px = ox + (c + 0.5) * res
                py = oy + (r + 0.5) * res
                if L2 < 1e-9:
                    d = math.hypot(px - p1[0], py - p1[1])
                else:
                    t = max(0.0, min(1.0,
                        ((px - p1[0]) * dx + (py - p1[1]) * dy) / L2))
                    fx = p1[0] + t * dx
                    fy = p1[1] + t * dy
                    d = math.hypot(px - fx, py - fy)
                if d <= infl:
                    occ[r, c] = 1

    def _plan_path(self, start_xy: Tuple[float, float],
                   goal_xy:  Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        8-connected A* on the inflated occupancy grid.  Returns a list of
        world-frame waypoints from start to goal inclusive.  If there is
        no floor_map, returns a straight line.  If start or goal is in an
        inflated cell, the nearest free cell is used.
        """
        occ_data = self._build_occupancy()
        if occ_data is None:
            return [start_xy, goal_xy]
        occ, origin, res = occ_data
        H, W = occ.shape

        def to_cell(xy):
            return (int((xy[1] - origin[1]) / res),
                    int((xy[0] - origin[0]) / res))

        def to_world(cell):
            return (origin[0] + (cell[1] + 0.5) * res,
                    origin[1] + (cell[0] + 0.5) * res)

        def nearest_free(cell):
            """If cell is occupied, radiate outward to find the closest free cell."""
            r0, c0 = cell
            if (0 <= r0 < H and 0 <= c0 < W and not occ[r0, c0]):
                return cell
            for radius in range(1, max(H, W)):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue
                        r, c = r0 + dr, c0 + dc
                        if 0 <= r < H and 0 <= c < W and not occ[r, c]:
                            return (r, c)
            return cell   # no free cell exists; degenerate

        s = nearest_free(to_cell(start_xy))
        g = nearest_free(to_cell(goal_xy))
        if s == g:
            return [start_xy, goal_xy]

        def heur(p):
            return math.hypot(p[0] - g[0], p[1] - g[1])

        heap = [(heur(s), 0.0, s)]
        came = {}
        gsc  = {s: 0.0}
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while heap:
            _, g_cur, cur = heapq.heappop(heap)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                world_path = [start_xy] + \
                             [to_world(c) for c in path] + [goal_xy]
                return self._simplify_path(world_path, occ, origin, res)
            if g_cur > gsc.get(cur, math.inf):
                continue
            for dr, dc in dirs:
                nb = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nb[0] < H and 0 <= nb[1] < W):
                    continue
                if occ[nb[0], nb[1]]:
                    continue
                step = math.hypot(dr, dc)
                ng   = gsc[cur] + step
                if ng < gsc.get(nb, math.inf):
                    came[nb] = cur
                    gsc[nb]  = ng
                    heapq.heappush(heap, (ng + heur(nb), ng, nb))

        # Unreachable: fall back to straight line
        return [start_xy, goal_xy]

    def _simplify_path(self, path, occ, origin, res):
        """
        Line-of-sight path shortening: walk from start, skip intermediate
        points when the straight segment to a later point clears all cells.
        Keeps endpoints.  Reduces staircase artefacts from grid A*.
        """
        if len(path) <= 2:
            return path

        def clear(a, b):
            """Bresenham-ish check along segment a->b against occ."""
            steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (res / 2)))
            for i in range(steps + 1):
                t = i / steps
                x = a[0] + t * (b[0] - a[0])
                y = a[1] + t * (b[1] - a[1])
                r = int((y - origin[1]) / res)
                c = int((x - origin[0]) / res)
                if not (0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]):
                    return False
                if occ[r, c]:
                    return False
            return True

        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not clear(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    # ==================================================================
    #  SECTION 1b - HOLONOMIC DRIVING ALONG A POLYLINE
    # ==================================================================

    def move_to(self, target_pose: Tuple[float, float, float],
                viz_label: str = "") -> dict:
        """
        Drive the base from self.pose to target_pose while avoiding any
        tables/walls declared in floor_map.

        Steps:
          1. Plan an obstacle-free polyline with A* on the inflated
             occupancy grid (SECTION 1).
          2. Shorten via line-of-sight simplification.
          3. Follow the polyline holonomically at linear_speed, spreading
             the heading change dth uniformly over the traversal time.
          4. Sample at dt and render motion_frames evenly spaced frames
             if self.visualize.

        With no floor_map, degenerates to the straight-line motion used
        previously.
        """
        x0, y0, th0 = self.pose
        xg, yg, thg = target_pose

        # --- 1-2. Plan obstacle-free path ---
        planned = self._plan_path((x0, y0), (xg, yg))
        self._last_planned_path = planned

        # Segment lengths and cumulative distance along the polyline
        seg_d = []
        for i in range(len(planned) - 1):
            ax, ay = planned[i]
            bx, by = planned[i + 1]
            seg_d.append(math.hypot(bx - ax, by - ay))
        total_dist = sum(seg_d)

        dth = self._wrap_angle(thg - th0)
        t_lin = total_dist / self.linear_speed  if self.linear_speed  > 0 else 0.0
        t_ang = abs(dth) / self.angular_speed   if self.angular_speed > 0 else 0.0
        T = max(t_lin, t_ang, 1e-6)   # avoid div-by-0 when already at target

        n_steps = max(1, int(math.ceil(T / self.dt)))

        # Pick which trajectory samples to render
        render_set = set()
        if self.visualize and self.motion_frames > 0:
            m = min(self.motion_frames, n_steps)
            stride = max(1, n_steps // m)
            render_set = {k for k in range(stride, n_steps + 1, stride)}
            render_set.add(n_steps)

        # --- 3-4. Follow the polyline ---
        def point_at(arclen):
            """Return (x, y) at a given distance along planned polyline."""
            if total_dist <= 1e-9:
                return planned[-1]
            remaining = arclen
            for i, d in enumerate(seg_d):
                if remaining <= d or i == len(seg_d) - 1:
                    local = remaining / d if d > 1e-9 else 0.0
                    local = max(0.0, min(1.0, local))
                    ax, ay = planned[i]
                    bx, by = planned[i + 1]
                    return (ax + local * (bx - ax),
                            ay + local * (by - ay))
                remaining -= d
            return planned[-1]

        for k in range(1, n_steps + 1):
            frac = k / n_steps
            xk, yk = point_at(frac * total_dist)
            thk    = self._wrap_angle(th0 + frac * dth)
            self.trajectory.append((xk, yk, thk))
            self.pose = (xk, yk, thk)
            if k in render_set:
                self._viz_floor(label=viz_label or "moving")

        self.pose = tuple(target_pose)
        return {
            'from': (x0, y0, th0),
            'to': tuple(target_pose),
            'distance': total_dist,
            'rotation': dth,
            'travel_time': T,
            'samples': n_steps,
            'waypoints': planned,
        }

    # ==================================================================
    #  SECTION 2 - GBNN+H EXECUTION AT AN ACCESS POINT
    # ==================================================================

    def execute_at_ap(self, ap: AccessPoint) -> dict:
        """
        Build a GBNN_H planner for this access point's RoI, run it to
        completion (or step_cap), and return the planner output plus a
        `complete` flag based on completion_threshold.

        If self.visualize and self.show_gbnn are both True, the GBNN+H
        run is rendered with its own per-step viz (fresh figure per
        step).  This naturally interleaves with the floor animation
        from move_to.
        """
        kw = dict(self.gbnn_kwargs)
        kw.update(ap.gbnn_kwargs)
        if self.visualize and self.show_gbnn:
            kw['visualize'] = True
        elif 'visualize' not in kw:
            kw['visualize'] = False
        planner = GBNN_H(ap.roi_grid.copy(), **kw)
        paths, stats = planner.run()
        complete = stats['coverage'] >= self.completion_threshold
        return {
            'ap_label': ap.label,
            'ap_pose': ap.pose,
            'paths': paths,
            'stats': stats,
            'complete': complete,
        }

    # ==================================================================
    #  SECTION 3 - MAIN SEQUENCING LOOP
    # ==================================================================

    def run(self) -> List[dict]:
        """
        Attend to every access point in order.  Per AP:

            1. move_to(ap.pose)
            2. execute_at_ap(ap) -> wait for complete flag
            3. if not complete and retries remain, re-run GBNN+H
            4. advance to next AP

        Returns the per-AP log.  The sequence only ends once every AP
        has been visited; APs that fail to complete after all retries
        are recorded as `complete=False` but do not abort the sweep.
        """
        t0 = time.time()
        if self.verbose:
            print(f"[MobilePlatform] Starting sweep of "
                  f"{len(self.access_points)} access point(s) "
                  f"from pose {self.pose}")

        # Initial overview frame: floor + all APs + robot at start
        if self.visualize:
            self._viz_floor(label="initial")

        for i, ap in enumerate(self.access_points):
            tag = ap.label or f"AP{i}"
            if self.verbose:
                print(f"\n[MobilePlatform] ({i+1}/{len(self.access_points)}) "
                      f"-> {tag}  pose={ap.pose}")

            motion = self.move_to(ap.pose, viz_label=f"-> {tag}")

            # Arrival frame: robot at AP, before GBNN+H fires
            if self.visualize:
                self._viz_floor(label=f"arrived at {tag}, starting GBNN+H")
            if self.verbose:
                print(f"[MobilePlatform] moved {motion['distance']:.2f} u, "
                      f"rotated {math.degrees(motion['rotation']):+.1f} deg, "
                      f"travel_time={motion['travel_time']:.2f}s")

            attempt = 0
            while True:
                result = self.execute_at_ap(ap)
                attempt += 1
                if result['complete']:
                    break
                if attempt > self.max_retries:
                    if self.verbose:
                        print(f"[MobilePlatform]  ! {tag}: not complete after "
                              f"{attempt} attempt(s), coverage="
                              f"{result['stats']['coverage']:.3f}")
                    break
                if self.verbose:
                    print(f"[MobilePlatform]  retry {attempt}/"
                          f"{self.max_retries} at {tag} "
                          f"(coverage={result['stats']['coverage']:.3f})")

            result['attempts'] = attempt
            self.ap_log.append(result)

            if self.verbose:
                flag = "OK" if result['complete'] else "PARTIAL"
                print(f"[MobilePlatform] {tag} {flag}: "
                      f"steps={result['stats']['steps']}  "
                      f"flight={result['stats']['flight']}  "
                      f"coverage={result['stats']['coverage']:.3f}")

        wall = time.time() - t0
        done = sum(1 for r in self.ap_log if r['complete'])
        if self.verbose:
            print(f"\n[MobilePlatform] Sweep complete: "
                  f"{done}/{len(self.ap_log)} APs reached threshold  "
                  f"wall={wall:.2f}s")
        return self.ap_log

    # ==================================================================
    #  SECTION 4 - TOP-DOWN VISUALISATION (floor view)
    # ==================================================================

    def _floor_bounds(self):
        """Return (xmin, ymin, xmax, ymax) for the floor plot."""
        if self.floor_map is not None:
            return self.floor_map.bounds
        # Infer from AP poses + robot pose with a small margin
        xs = [p[0] for p in self.trajectory] + \
             [ap.pose[0] for ap in self.access_points]
        ys = [p[1] for p in self.trajectory] + \
             [ap.pose[1] for ap in self.access_points]
        pad = 1.0
        return (min(xs) - pad, min(ys) - pad,
                max(xs) + pad, max(ys) + pad)

    def _draw_robot(self, ax, x, y, th, size=0.35):
        """Draw MDA base as a triangle pointing along theta."""
        # Lazy import — _draw_robot is only called from _viz_floor, which
        # has already imported matplotlib in its own scope.
        import matplotlib.patches as patches

        nose = (x + size * math.cos(th),
                y + size * math.sin(th))
        a    = size * 0.7
        lft  = (x + a * math.cos(th + 2.4),
                y + a * math.sin(th + 2.4))
        rgt  = (x + a * math.cos(th - 2.4),
                y + a * math.sin(th - 2.4))
        ax.add_patch(patches.Polygon(
            [nose, lft, rgt],
            facecolor='tab:orange', edgecolor='black',
            linewidth=1.2, zorder=8))

    def _viz_floor(self, label=""):
        """
        Top-down floor frame.  Reuses a single persistent figure window
        (named "GBNN+H Floor") across calls; each call clears and redraws
        in place via plt.pause.
        Renders:
          - floor extent + optional gridlines
          - tables / walls from self.floor_map
          - access points (green=complete, grey=visited partial,
                           red=pending, with heading arrow)
          - trajectory trail
          - current robot pose as an orange triangle
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.ion()
        fig = plt.figure(num='GBNN+H Floor', figsize=(9, 9))
        fig.clf()
        ax = fig.add_subplot(111)
        xmin, ymin, xmax, ymax = self._floor_bounds()

        # --- Floor background ---
        ax.add_patch(patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor='#F2EFEA', edgecolor='black', linewidth=1.5,
            zorder=0))

        # --- Gridlines ---
        if self.floor_map is not None and self.floor_map.grid_step > 0:
            step = self.floor_map.grid_step
            x = xmin
            while x <= xmax:
                ax.plot([x, x], [ymin, ymax], color='#D0CEC9',
                        linewidth=0.4, zorder=1)
                x += step
            y = ymin
            while y <= ymax:
                ax.plot([xmin, xmax], [y, y], color='#D0CEC9',
                        linewidth=0.4, zorder=1)
                y += step

        # --- Tables / walls ---
        if self.floor_map is not None:
            for (tx, ty, tw, th_) in self.floor_map.tables:
                ax.add_patch(patches.Rectangle(
                    (tx, ty), tw, th_,
                    facecolor='#8B6F47', edgecolor='black',
                    linewidth=1.0, zorder=2))
            for (p1, p2) in self.floor_map.walls:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color='black', linewidth=2.5, zorder=2)

        # --- Trajectory trail (actual) + current planned leg ---
        if len(self.trajectory) > 1:
            xs = [p[0] for p in self.trajectory]
            ys = [p[1] for p in self.trajectory]
            ax.plot(xs, ys, '-', color='tab:blue',
                    linewidth=1.2, alpha=0.7, zorder=3,
                    label='trajectory')
        if len(self._last_planned_path) > 1:
            pxs = [p[0] for p in self._last_planned_path]
            pys = [p[1] for p in self._last_planned_path]
            ax.plot(pxs, pys, '--', color='tab:purple',
                    linewidth=1.0, alpha=0.8, zorder=4,
                    label='planned leg')

        # --- Access points ---
        n_done = len(self.ap_log)
        for i, ap in enumerate(self.access_points):
            x, y, thap = ap.pose
            if i < n_done:
                col = 'tab:green' if self.ap_log[i]['complete'] \
                      else 'tab:gray'
            else:
                col = 'tab:red'
            ax.scatter(x, y, c=col, s=180, edgecolors='black',
                       linewidths=1.2, zorder=5)
            ax.arrow(x, y,
                     0.35 * math.cos(thap), 0.35 * math.sin(thap),
                     head_width=0.12, head_length=0.12,
                     fc=col, ec='black', linewidth=0.6, zorder=6)
            ax.annotate(ap.label or f'AP{i}', (x, y),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        # --- Robot current pose ---
        self._draw_robot(ax, self.pose[0], self.pose[1], self.pose[2])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'MobilePlatform | {label} | '
                     f'AP {n_done}/{len(self.access_points)} done')
        ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(0.1)

    def plot_path(self):
        """Convenience: render a single final-state floor frame."""
        self._viz_floor(label="final")


# ============================================================================
#  UTILITIES
# ============================================================================

def make_grid(H, V, obstacle_chance=0.05, seed=None):
    """Generate a random H x V grid.  -1.0 = obstacle, 1.0 = free."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(H, V),
                      p=[obstacle_chance, 1.0 - obstacle_chance])


# Reference test grids (from gbnn_h.py)
REF_GRIDS = {
    1: np.ones((12, 12), dtype=float),

    2: np.array([
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1],
    ], dtype=float),

    3: np.array([
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1],
        [-1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1],
        [ 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1],
        [-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1],
    ], dtype=float),
}


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # --- Demo A: standalone GBNN_H on one RoI, live rendering ---
    #
    # grid = REF_GRIDS[3].copy()
    # GBNN_H(grid, n_ee=2, visualize=True, viz_interval=1,
    #        step_cap=2000).run()

    # --- Demo B: MobilePlatform sweeping three APs with full viz ---
    #
    # Flow: [overview] -> animate base moving toward AP1 -> [arrival]
    #       -> GBNN+H per-step viz at AP1 -> animate to AP2 -> ...
    #
    # Each matplotlib window is its own frame: close
    # one to advance.  Set visualize=False to run headless.

    # Hospital ward layout: 10 x 8 floor with a bed and a side table
    floor = FloorMap(
        bounds=(0.0, 0.0, 10.0, 8.0),
        tables=[
            (2.0, 4.5, 3.0, 1.8),   # hospital bed
            (6.5, 2.5, 1.5, 1.0),   # side table
        ],
        walls=[
            ((0.0, 0.0), (10.0, 0.0)),
            ((10.0, 0.0), (10.0, 8.0)),
            ((10.0, 8.0), (0.0, 8.0)),
            ((0.0, 8.0), (0.0, 0.0)),
        ],
        grid_step=1.0,
    )

    access_points = [
        AccessPoint(pose=(2.0, 3.8, math.pi / 2),
                    roi_grid=REF_GRIDS[1].copy(),
                    label="bed_head"),
        AccessPoint(pose=(5.5, 5.4, 0.0),
                    roi_grid=REF_GRIDS[2].copy(),
                    label="bed_side"),
        AccessPoint(pose=(7.2, 1.8, -math.pi / 2),
                    roi_grid=REF_GRIDS[3].copy(),
                    label="side_table"),
    ]

    mdp = MobilePlatform(
        access_points=access_points,
        initial_pose=(0.5, 0.5, 0.0),
        linear_speed=0.8,
        angular_speed=1.0,
        floor_map=floor,
        visualize=True,       # floor + GBNN+H animation
        motion_frames=6,      # frames per leg of the trip
        show_gbnn=True,       # show per-step GBNN+H viz at each AP
        gbnn_kwargs=dict(n_ee=2, viz_interval=5, step_cap=2000),
        verbose=True,
    )
    log = mdp.run()

    print("\n--- Summary ---")
    for i, r in enumerate(log):
        flag = "OK" if r['complete'] else "PARTIAL"
        print(f"  {i+1}. {r['ap_label']:<12s} {flag}  "
              f"steps={r['stats']['steps']:4d}  "
              f"flight={r['stats']['flight']}  "
              f"cov={r['stats']['coverage']:.3f}")