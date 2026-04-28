# common — Shared utilities

This folder contains shared modules imported by `demo.py` and the algorithm modules.

## Contents

- **[`obstacles.py`](obstacles.py)** — Obstacle and collision system for the demo sandbox. Defines `ObstacleKind` (table, chair, walls, pillars, sliding/pivot doors, trolleys, humans), the `ObstacleManager`, the `OccupancyGrid`, and pathfinding helpers (`pathfind_astar`, `pathfind_dijkstra`, `smooth_path`).
- **[`replicated_gbnn.py`](replicated_gbnn.py)** — Reference re-implementation of the Glasius Bioinspired Neural Network (Glasius, Komoda & Gielen, 1995). GBNN is **not** a contribution of this codebase; it is included as the prior-art baseline for the GBNN+H derivative. See the file's header docstring for the full attribution.

## Note on `HEAVY_TROLLEY`

The `HEAVY_TROLLEY` obstacle kind is exposed in the demo's user interface (toolbar and spawn key) but is **not fully functional in this release**. Its complete behaviour — including the centroid-pin rotation override and full fusion / fission semantics — will land alongside upcoming code contributions in a future release.

Testers can spawn and interact with `HEAVY_TROLLEY` instances but should be aware that some interactions may be incomplete or stubbed out compared to `LOW_TROLLEY` and `HIGH_TROLLEY`.
