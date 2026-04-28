# GBNN+H — Hierarchical Graph-Based Neural Network for Dual-Arm Coverage

> **Algorithm:** Hierarchical GBNN extension for dual-arm mobile-manipulator coverage planning.
> **Headline:** Heuristic-driven runtime speedup over the baseline Glasius GBNN, while preserving complete area coverage.
> Built on the Glasius GBNN base (see [`../common/replicated_gbnn.py`](../common/replicated_gbnn.py)).

## Demo

![GBNN+H hierarchical coverage with dual-arm mobile manipulator](../assets/gbnnh_demo.gif)

## Script

[`open_gbnnh.py`](open_gbnnh.py) — pure Python class `GBNN_H`.

```python
from gbnnh.open_gbnnh import GBNN_H
planner = GBNN_H(...)
```

## Paper

**Title:** *Complete Area-Coverage Path Planner for Surface Cleaning in Hospital Settings Using Mobile Dual-Arm Robot and GBNN with Heuristics*
**Authors:** Ash Yaw Sang Wan, Lim Yi, Abdullah Aamir Hayat, Chee Gen Moo, Mohan Rajesh Elara
**Venue:** [Springer *Complex & Intelligent Systems*](https://link.springer.com/article/10.1007/s40747-024-01483-3), vol. 10, no. 5, pp. 6767–6785, 2024
**DOI:** [10.1007/s40747-024-01483-3](https://doi.org/10.1007/s40747-024-01483-3)

## Cite

```bibtex
@article{wan2024complete,
  title   = {Complete area-coverage path planner for surface cleaning in hospital settings using mobile dual-arm robot and GBNN with heuristics},
  author  = {Wan, Ash Yaw Sang and Yi, Lim and Hayat, Abdullah Aamir and Gen, Moo Chee and Elara, Mohan Rajesh},
  journal = {Complex \& Intelligent Systems},
  volume  = {10},
  number  = {5},
  pages   = {6767--6785},
  year    = {2024},
  publisher = {Springer},
  doi     = {10.1007/s40747-024-01483-3}
}
```

## Funding

This research was supported by the National Robotics Programme under its National Robotics Programme (NRP) BAU, *Ermine III: Deployable Reconfigurable Robots*, Award No. **M22NBK0054**, and by A\*STAR under its *RIE2025 IAF-PP — Advanced ROS2-native Platform Technologies for Cross-sectorial Robotics Adoption* programme, Award No. **M21K1a0104** (National Robotics Program Grant No. **1922200058**).
