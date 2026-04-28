#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interstar.py  —  Inter-Star: Modified Multi-Robot A* Path Planning
===================================================================
Paper : "Inter-Star: A Modified Multi A-Star Approach for
         Inter-Reconfigurable Robots"
        Expert Systems with Applications, 2025, p.129134
Thesis: Ch.4 — Towards Computationally Scalable Inter-Reconfigurable Robots
Author: Ash Wan Yaw Sang  |  SUTD 2025  |  ROAR Lab

Algorithm overview
------------------
Standard multi-agent A* runs one independent search per robot → O(n) total
node expansions.  Inter-Star exploits the shared path segment that arises
when multiple robots converge to one goal (fusion) or diverge from one
start (fission).  The first robot computes the shared segment in full; every
subsequent robot short-circuits the moment its frontier reaches a cell already
claimed by an earlier robot, then stitches its path onto the shared tail.
Measured gain: ≥ 3.02× fewer expansions vs. independent A*.

Grid encoding
-------------
    0 = free cell
    1 = obstacle

Modes
-----
fusion  — n robots, distinct starts, one shared goal.
          Inferred when goal is a single (row, col).
          output: path[k] = [start_k, ..., goal]

fission — one robot, one fission start, n distinct goals.
          Inferred when goal is a list of (row, col).
          Internally each search runs reversed (goal_k → fission_start).
          A join-point post-process trims reverse-motion detours.
          output: path[k] = [fission_start, ..., goal_k]

Usage
-----
    from Interstar import Interstar, sim1_grid

    grid  = sim1_grid(size=25)
    sim   = Interstar(n_robots=10, grid=grid, goal=(11, 6))
    paths = sim.run(show_search=True)
    sim.visualize()
"""

import heapq
import math
import time
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
#  UTILITIES
# ============================================================================

def sim1_grid(size=25):
    """
    Simulation 1.1 — random square grid with ~5% obstacle density.
    Returns a 2-D list (0 = free, 1 = obstacle).
    """
    from random import randint
    return [[1 if randint(0, 100) > 95 else 0
             for _ in range(size)]
            for _ in range(size)]


# ============================================================================
#  Interstar
# ============================================================================

class Interstar:
    """
    Inter-Star multi-robot A* path planner.

    Parameters
    ----------
    n_robots : int
        Number of robots.  For fission, overridden by len(goal).
    grid     : 2-D list  (0 = free, 1 = obstacle)
        Environment map — any shape.
    goal     : (row, col) or [(row,col), ...]
        Single point  → fusion  (n robots → 1 goal).
        List of points → fission (1 random start → n goals).
    """

    COLORS = ['b', 'g', 'r', 'c', 'm', 'k',
              'orange', 'purple', 'brown', 'pink']

    _DIRS = [(0,-1),(0,1),(-1,0),(1,0),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_robots,
        grid,
        goal,
        mode=None,
        starts=None,
        fission_start=None,
    ):
        """
        Parameters
        ----------
        n_robots : int
            Number of robots (ignored for fission — overridden by len(goal)).
        grid : 2-D list OR numpy array  (0 = free, 1 = obstacle)
            Environment map.  Numpy input is converted to a nested list so
            the existing index-by-index traversal still works.
        goal : (row, col) or [(row,col), ...]
            Single point  → fusion   (n robots → 1 goal).
            List of points → fission (1 start → n goals).
        mode : None / "fusion" / "fission", optional
            Force the mode.  If None (default), mode is inferred from the
            shape of `goal` (scalar tuple = fusion, list of tuples = fission).
        starts : list of (row, col), optional
            Fusion only.  Explicit robot start positions in grid coords.
            If None (default), random free cells are picked — keeps the
            standalone matplotlib-demo behaviour intact.
        fission_start : (row, col), optional
            Fission only.  Explicit divergence-start position.
            If None (default), random free cell is picked.
        """
        # Accept numpy arrays — adapters (demo.py) rasterise a numpy grid.
        # The core algorithm still indexes as `grid[r][c]`, so convert once.
        if hasattr(grid, "tolist") and callable(getattr(grid, "tolist")):
            grid = grid.tolist()
        self.grid = [list(row) for row in grid]   # local copy

        # Mode: explicit `mode` arg wins; else infer from goal shape.
        if mode is not None:
            if mode not in ("fusion", "fission"):
                raise ValueError(
                    f"mode must be 'fusion' or 'fission' (got {mode!r})")
            self.mode = mode
        elif isinstance(goal[0], (int, float)):
            self.mode = "fusion"
        else:
            self.mode = "fission"

        if self.mode == "fusion":
            self.goal = [int(goal[0]), int(goal[1])]
            self.fission_goals = None
            self.fission_start = None
        else:
            self.fission_goals = [[int(g[0]), int(g[1])] for g in goal]
            self.goal = None
            self.fission_start = (
                [int(fission_start[0]), int(fission_start[1])]
                if fission_start is not None else None
            )

        self.n_robots = n_robots if self.mode == "fusion" else len(goal)

        # Explicit starts for fusion (adapter path); None → random via
        # _generate_starts (standalone-demo path).
        if starts is not None:
            if self.mode != "fusion":
                raise ValueError(
                    "`starts` only applies to fusion mode.")
            self.starts = [[int(s[0]), int(s[1])] for s in starts]
        else:
            self.starts = None

        # Set by run()
        self.W     = None   # refmap / success grid: W[r][c] = robot index or inf
        self.P     = None   # raw paths (end-first, both endpoints)
        self.paths = None   # final output paths (start-first)

        self._generate_starts()

    # ==================================================================
    #  SECTION 1 — MAP INITIALISATION
    # ==================================================================

    def _generate_starts(self):
        """
        Populate robot start positions on free cells of the provided grid.

        Fusion  : uses `self.starts` if set by caller; else generates
                  n_robots random free positions.  Forces goal cell free.
        Fission : uses `self.fission_start` if set by caller; else picks
                  one random free cell.  Forces all goal cells free.
        """
        from random import randint
        rows = len(self.grid)
        cols = len(self.grid[0])

        def random_free():
            while True:
                r, c = randint(0, rows - 1), randint(0, cols - 1)
                if self.grid[r][c] == 0:
                    return [r, c]

        if self.mode == "fusion":
            if self.starts is None:
                self.starts = [random_free() for _ in range(self.n_robots)]
            self.grid[self.goal[0]][self.goal[1]] = 0
        else:
            if self.fission_start is None:
                self.fission_start = random_free()
            for g in self.fission_goals:
                self.grid[g[0]][g[1]] = 0

    # ==================================================================
    #  SECTION 2 — LIVE SEARCH VISUALISATION
    # ==================================================================

    def _viz(self, distances, start, end, current=None):
        """
        Inferno heatmap of g-costs.  Reuses a single persistent figure
        window (named "Inter-Star search") across iterations: each call
        clears and redraws on the same window for smooth in-place updates.

        distances : 2-D g-cost grid  (inf = unvisited → black)
        start     : [row, col] of this robot's search start
        end       : [row, col] of the search target
        current   : (row, col) node just updated — lime square
        """
        plt.ion()
        dist_arr = np.array(distances, dtype=float)
        dist_arr[~np.isfinite(dist_arr)] = np.nan

        fig = plt.figure(num='Inter-Star search', figsize=(6, 6))
        fig.clf()
        ax = fig.add_subplot(111)

        cmap = plt.cm.inferno
        cmap.set_bad(color='black')
        ax.imshow(dist_arr, cmap=cmap, origin='upper')

        ax.scatter(start[1], start[0], c='cyan', marker='o', s=60,
                   label='Start')
        ax.scatter(end[1],   end[0],   c='red',  marker='X', s=60,
                   label='End')

        # White dots — cells already claimed by earlier robots
        if self.W is not None:
            succ_arr  = np.array(self.W, dtype=float)
            used_mask = np.isfinite(succ_arr)
            ys, xs    = np.where(used_mask)
            if len(xs):
                ax.scatter(xs, ys, c='white', marker='o', s=15, alpha=0.7)

        if current is not None:
            cx, cy = current
            ax.scatter(cy, cx, c='lime', marker='s', s=80, label='Current')

        ax.set_title("Inter-Star search distances (live)")
        ax.legend(loc='upper right')
        fig.canvas.draw_idle()
        plt.pause(0.1)

    # ==================================================================
    #  SECTION 3 — PATH STITCHING  (Algorithm 2 — Traceline)
    # ==================================================================

    def _trace(self, node, P_ref, predecessors):
        """
        Patch `predecessors` so reconstruction from the search-end follows
        P_ref down to `node`, then transitions into the current robot's own
        predecessor chain.

        Implements Algorithm 2 (Traceline) from the paper.

        node        : [row, col] — shared cell where stitching occurs
        P_ref       : reference robot's raw path, end-first ([end, ..., start])
        predecessors: current robot's predecessor map — mutated in place
        """
        traced = []
        for wp in P_ref:
            if wp == node:                                    # Alg. 2 line 3
                break
            if traced:                                        # Alg. 2 line 4-5
                predecessors[traced[-1][0]][traced[-1][1]] = (wp[0], wp[1])
            traced.append(wp)
        if traced:                                            # Alg. 2 line 8
            predecessors[traced[-1][0]][traced[-1][1]] = (node[0], node[1])

    # ==================================================================
    #  SECTION 4 — COST FUNCTIONS  (Equations 2 and 3)
    # ==================================================================

    @staticmethod
    def _h(node, goal):
        """
        Admissible heuristic: Euclidean distance (Eqn. 3).
            h(i) = sqrt( sum_{i in {x,y}} (C_i - G_i)^2 )
        """
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    @staticmethod
    def _step_cost(dx, dy):
        """Movement cost for one diagonal or cardinal step."""
        return math.sqrt(dx*dx + dy*dy)

    # ==================================================================
    #  SECTION 5 — CORE INTER-STAR SEARCH  (Algorithm 1)
    # ==================================================================

    # Tracked across all _run_search calls during a single run() invocation
    # — used by the demo.py adapter to report expansions-vs-baseline metric.
    _expansions_this_run: int = 0

    def _run_search(self, S_k, G, k, show_search=False):
        """
        Single Inter-Star A* search for robot k from S_k to G.

        Implements Algorithm 1 from the paper.
        Mutates self.W (claims cells) and appends the raw path to self.P.
        Raw path is stored end-first and includes both endpoints.

        Parameters
        ----------
        S_k         : [row, col]  robot k's search start
        G           : [row, col]  search target (Alg. 1: end)
        k           : robot index
        show_search : call _viz on every neighbour update if True
        """
        grid = self.grid
        rows, cols = len(grid), len(grid[0])

        def is_valid(x, y):
            return 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0

        # g  : true path cost (Alg. 1: g(n))
        # costmap : f-value C = g + h (Alg. 1: costmap, Eqn. 2)
        g       = [[float('inf')] * cols for _ in range(rows)]
        costmap = [[float('inf')] * cols for _ in range(rows)]
        g[S_k[0]][S_k[1]]       = 0
        costmap[S_k[0]][S_k[1]] = 0

        predecessors = [[None] * cols for _ in range(rows)]
        pq           = [(0, (S_k[0], S_k[1]))]   # (f, node) — visited heap

        while pq:                                             # Alg. 1 line 3
            f_val, (cx, cy) = heapq.heappop(pq)              # Alg. 1 line 4
            self._expansions_this_run += 1

            if (cx, cy) == (G[0], G[1]):                     # Alg. 1 line 5
                break

            if self.W[cx][cy] != float('inf'):               # Alg. 1 line 7
                ref_k = int(self.W[cx][cy])
                self._trace([cx, cy], self.P[ref_k],         # Alg. 1 line 8-9
                            predecessors)
                break                                         # Alg. 1 line 10

            for dx, dy in self._DIRS:                        # Alg. 1 line 11
                nx, ny = cx + dx, cy + dy                    # Alg. 1 line 12
                if is_valid(nx, ny):                         # Alg. 1 line 13
                    g_child = g[cx][cy] + self._step_cost(dx, dy)
                    f_child = g_child   + self._h((nx, ny), G)  # Eqn. 2
                    if costmap[nx][ny] > f_child:            # Alg. 1 line 15
                        costmap[nx][ny]      = f_child       # Alg. 1 line 16
                        g[nx][ny]            = g_child
                        predecessors[nx][ny] = (cx, cy)      # Alg. 1 line 17
                        heapq.heappush(pq, (f_child, (nx, ny)))  # Alg. 1 line 18
                        if show_search:
                            self._viz(g, S_k, G, current=(nx, ny))

        # Reconstruct path: walk from G via predecessors until None
        # Alg. 1 line 19-20: path = path + traceline; append refmap
        path    = []
        current = (G[0], G[1])
        while current is not None:
            path.append([current[0], current[1]])
            r, c    = current
            nxt     = predecessors[r][c]
            current = (nxt[0], nxt[1]) if nxt is not None else None

        for node in path:                                     # Alg. 1 line 20
            self.W[node[0]][node[1]] = k

        self.P.append(path)

    # ==================================================================
    #  SECTION 6 — FISSION JOIN-POINT  (§4.5)
    # ==================================================================

    def _find_join_point(self, path, S, G_k):
        """
        Return index in `path` (fission_start-first) that minimises
            dist(wp, S) + dist(wp, G_k).

        The fleet layer navigates S → path[j] directly; the robot then
        follows path[j:] onward to G_k, avoiding reverse-motion detours.
        """
        best_i, best_score = 0, float('inf')
        for i, wp in enumerate(path):
            score = (self._h(wp, S) + self._h(wp, G_k))
            if score < best_score:
                best_score = score
                best_i     = i
        return best_i

    # ==================================================================
    #  SECTION 7 — PUBLIC API
    # ==================================================================

    def run(self, show_search=False):
        """
        Execute the Inter-Star algorithm (Algorithm 1, outer loop).

        Steps
        -----
        1. Initialise W (refmap) and P (paths).
        2. For each robot k, run _run_search from S_k to G.
        3. Reconstruct final paths in travel order (start → goal).
           Fusion  : reverse raw paths (stored end-first).
           Fission : paths already start-first; apply join-point trim.

        Parameters
        ----------
        show_search : bool — live inferno heatmap on every neighbour update

        Returns
        -------
        self.paths : list of n paths.
            Each path is a list of [row, col] in travel order.
            Returns [] for robot k if no path found.
        """
        rows, cols = len(self.grid), len(self.grid[0])
        self.W = [[float('inf')] * cols for _ in range(rows)]
        self.P = []
        self._expansions_this_run = 0   # reset per-run counter

        # ── Fusion ──────────────────────────────────────────────────────
        if self.mode == 'fusion':
            G = self.goal
            for k, S_k in enumerate(self.starts):
                self._run_search(S_k, G, k, show_search)

            # P[k] = [G, ..., S_k]  (G-first) → reverse → start-first
            self.paths = [list(reversed(raw)) for raw in self.P]

        # ── Fission ─────────────────────────────────────────────────────
        else:
            S     = self.fission_start
            goals = self.fission_goals

            # Run each search reversed: goal_k → fission_start (S)
            for k, G_k in enumerate(goals):
                self._run_search(G_k, S, k, show_search)

            # P[k] = [S, ..., G_k]  (S-first, both endpoints)
            self.paths = []
            for k, raw in enumerate(self.P):
                if not raw:
                    self.paths.append([])
                    continue
                G_k = goals[k]
                j   = self._find_join_point(raw, S, G_k)
                self.paths.append(raw if j == 0 else [S] + raw[j:])

        return self.paths

    # ==================================================================
    #  SECTION 8 — FINAL PATH VISUALISATION
    # ==================================================================

    def visualize(self, title=None):
        """
        Overlay computed paths on the grid using the cividis colormap.
        Must be called after run().
        """
        if self.paths is None:
            raise RuntimeError("Call run() before visualize().")

        if title is None:
            title = (f"Inter-Star — {self.mode.capitalize()} "
                     f"({self.n_robots} robots)")

        # Build display: obstacle = -1, free = 0, path = 1
        display = [[-1.0 if cell == 1 else 0.0 for cell in row]
                   for row in self.grid]
        for path in self.paths:
            for wp in path:
                display[wp[0]][wp[1]] = 1.0

        # Metabuffer border
        plotgraph  = np.array(display, dtype=float)
        nr, nc     = plotgraph.shape
        col_border = np.full((nr, 1), -1.0)
        plotgraph  = np.hstack([col_border, plotgraph, col_border])
        row_border = np.full((1, nc + 2), -1.0)
        plotgraph  = np.vstack([row_border, plotgraph, row_border])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(plotgraph, cmap=plt.cm.cividis, origin='upper')

        if self.mode == 'fusion':
            for k, s in enumerate(self.starts):
                ax.scatter(s[1] + 1, s[0] + 1,
                           marker='*', color=self.COLORS[k % len(self.COLORS)],
                           s=300)
            ax.scatter(self.goal[1] + 1, self.goal[0] + 1,
                       marker=6, color='red', s=500, label='Goal')
        else:
            ax.scatter(self.fission_start[1] + 1, self.fission_start[0] + 1,
                       marker='*', color='cyan', s=500, label='Fission start')
            for k, gk in enumerate(self.fission_goals):
                ax.scatter(gk[1] + 1, gk[0] + 1,
                           marker=6, color=self.COLORS[k % len(self.COLORS)],
                           s=300)

        plt.title(title, size=20)
        plt.grid(color='black', which='both', linewidth=1)
        plt.xticks(np.linspace(0.5, len(plotgraph[0]) - 0.5, len(plotgraph[0])))
        plt.yticks(np.linspace(0.5, len(plotgraph) - 0.5, len(plotgraph)))
        ax.legend(loc='upper right')
        plt.show()


    # ==================================================================
    #  SECTION 9 — REFERENCE A* BASELINE  (for scalability benchmarks)
    # ==================================================================

    def baseline_expansions(self, starts, goal):
        """
        Run independent standard A* for each (start → goal) pair and return
        the TOTAL node-expansion count.  This is the baseline Inter-Star
        claims to beat by ≥3.02× — the ratio is consumed by the demo.py
        adapter as the "Interstar: N expansions (X× vs A*)" HUD metric.

        No path reconstruction, no shared-path stitching — pure A* cost
        counting for direct comparison with `_expansions_this_run`.
        """
        rows, cols = len(self.grid), len(self.grid[0])
        total = 0
        for S in starts:
            g       = [[float("inf")] * cols for _ in range(rows)]
            costmap = [[float("inf")] * cols for _ in range(rows)]
            g[S[0]][S[1]]       = 0
            costmap[S[0]][S[1]] = 0
            pq = [(0, (S[0], S[1]))]
            while pq:
                _, (cx, cy) = heapq.heappop(pq)
                total += 1
                if (cx, cy) == (goal[0], goal[1]):
                    break
                for dx, dy in self._DIRS:
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < rows and 0 <= ny < cols
                            and self.grid[nx][ny] == 0):
                        g_child = g[cx][cy] + self._step_cost(dx, dy)
                        f_child = g_child + self._h((nx, ny), goal)
                        if costmap[nx][ny] > f_child:
                            costmap[nx][ny] = f_child
                            g[nx][ny]       = g_child
                            heapq.heappush(pq, (f_child, (nx, ny)))
        return total

    # ==================================================================
    #  SECTION 10 — ADAPTER CONVENIENCE API  (used by demo.py)
    # ==================================================================

    @classmethod
    def plan(
        cls,
        starts,
        goal,
        grid,
        mode=None,
        fission_start=None,
        render=False,
    ):
        """
        One-shot convenience entry point for the demo.py `InterstarPlanner`
        adapter.  Instantiates + runs + returns `(paths, metrics)`.

        Parameters
        ----------
        starts : list of (row, col)
            Fusion: robot start positions.
            Fission: ignored (use `fission_start` param).
        goal : (row, col) or [(row, col), ...]
            Single point = fusion, list = fission.
        grid : 2-D list or numpy array (0 = free, 1 = obstacle)
        mode : None / "fusion" / "fission" — force mode, else auto-infer
        fission_start : (row, col), optional — fission only
        render : bool — if False, no matplotlib calls are made

        Returns
        -------
        (paths, metrics) : tuple
            paths  : list of per-robot [row, col] polylines in travel order
            metrics: dict with keys
                        "expansions"          : Inter-Star total expansions
                        "baseline_expansions" : independent A* total
                        "expansions_ratio"    : baseline / interstar
                                                (>1.0 = Inter-Star is cheaper)
                        "shared_segment"      : longest-common-suffix among
                                                all paths (fusion mode only)
        """
        if mode is None:
            mode = "fusion" if isinstance(goal[0], (int, float)) else "fission"

        n = len(starts) if mode == "fusion" else len(goal)
        sim = cls(
            n_robots      = n,
            grid          = grid,
            goal          = goal,
            mode          = mode,
            starts        = list(starts) if mode == "fusion" else None,
            fission_start = fission_start,
        )
        paths = sim.run(show_search=render)

        # Metrics
        metrics = {"expansions": int(sim._expansions_this_run)}
        if mode == "fusion":
            baseline = sim.baseline_expansions(sim.starts, sim.goal)
            metrics["baseline_expansions"] = baseline
            metrics["expansions_ratio"] = (
                baseline / sim._expansions_this_run
                if sim._expansions_this_run > 0 else 1.0
            )
            # Shared segment: longest tail common to all fusion paths
            # (paths are start-first; shared segment is the tail).
            metrics["shared_segment"] = cls._longest_common_tail(paths)
        else:
            metrics["baseline_expansions"] = sim.baseline_expansions(
                sim.fission_goals, sim.fission_start)
            metrics["expansions_ratio"] = (
                metrics["baseline_expansions"] / sim._expansions_this_run
                if sim._expansions_this_run > 0 else 1.0
            )
            metrics["shared_segment"] = []
        return paths, metrics

    @staticmethod
    def _longest_common_tail(paths):
        """Return the longest suffix shared by every non-empty path."""
        non_empty = [p for p in paths if p]
        if len(non_empty) < 2:
            return []
        # Reverse-iterate each path in lockstep; stop when first mismatch.
        common = []
        min_len = min(len(p) for p in non_empty)
        for i in range(1, min_len + 1):
            wps = [p[-i] for p in non_empty]
            if all(wp == wps[0] for wp in wps):
                common.append(wps[0])
            else:
                break
        return list(reversed(common))


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    grid = sim1_grid(size=25)
    grid[11][6] = 0   # ensure goal cell is free

    # ── Fusion: single goal → mode inferred ─────────────────────────────────
    sim   = Interstar(n_robots=10, grid=grid, goal=(11, 6))
    t0    = time.time()
    paths = sim.run(show_search=True)
    print(f"Fusion  ({time.time()-t0:.3f}s)")
    for k, p in enumerate(paths):
        print(f"  Robot {k:2d}: {len(p):3d} wp  {p[0]} → {p[-1]}")
    sim.visualize()

    # ── Fission: list of goals → mode inferred ──────────────────────────────
    sim2  = Interstar(n_robots=10, grid=grid, goal=sim.starts)
    t0    = time.time()
    paths = sim2.run(show_search=True)
    print(f"\nFission ({time.time()-t0:.3f}s)")
    for k, p in enumerate(paths):
        print(f"  Robot {k:2d}: {len(p):3d} wp  {p[0]} → {p[-1]}")
    sim2.visualize()