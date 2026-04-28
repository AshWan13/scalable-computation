# Inter-Star — Modified Multi-A* for Docking / Splitting

> **Algorithm:** Modified Multi-A* path planner for inter-reconfigurable robots.
> **Complexity:** O(log n · x).

## Demo

![Inter-Star multi-robot path planning](../assets/interstar_demo.gif)

## Script

[`open_interstar.py`](open_interstar.py) — pure Python class `Interstar`.

```python
from interstar.open_interstar import Interstar
planner = Interstar(...)
```

## Paper

**Title:** *Inter-Star: A Modified Multi A-Star Approach for Inter-Reconfigurable Robots*
**Authors:** Ash Yaw Sang Wan, Yang Zhenyuan, Chee Gen Moo, M.A. Viraj J. Muthugala, Mohan Rajesh Elara
**Venue:** [Elsevier *Expert Systems with Applications*](https://www.sciencedirect.com/science/article/abs/pii/S0957417425027514), art. 129134, 2025
**DOI:** [10.1016/j.eswa.2025.129134](https://doi.org/10.1016/j.eswa.2025.129134)

## Cite

```bibtex
@article{wan2025inter,
  title   = {Inter-Star: A Modified Multi A-Star Approach for Inter-Reconfigurable Robots},
  author  = {Wan, Ash Yaw Sang and Zhenyuan, Yang and Gen, Moo Chee and Muthugala, M. A. Viraj J. and Elara, Mohan Rajesh},
  journal = {Expert Systems with Applications},
  pages   = {129134},
  year    = {2025},
  publisher = {Elsevier},
  doi     = {10.1016/j.eswa.2025.129134}
}
```

## Funding

This research was supported by the National Robotics Programme under its National Robotics Programme (NRP) BAU, *Ermine III: Deployable Reconfigurable Robots*, Award No. **M22NBK0054**, and by A\*STAR under its *RIE2025 IAF-PP — Modular Reconfigurable Mobile Robots (MR)²* programme, Grant No. **M24N2a0039**.
