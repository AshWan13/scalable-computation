# Configurer — Constant Complexity Framework

> **Algorithm:** FSM for fusion / fission control of inter-reconfigurable robots.
> **Complexity:** O(n) → O(1) per pairwise reconfiguration.

## Demo

![Configurer FSM in action — fusion / fission of inter-reconfigurable robots](../assets/configurer_demo.gif)

## Script

[`open_configurer.py`](open_configurer.py) — pure Python class `Configurer`.

```python
from configurer.open_configurer import Configurer
planner = Configurer(...)
```

## Paper

**Title:** *Enabling Framework for Constant Complexity Model in Autonomous Inter-Reconfigurable Robots*
**Authors:** Ash Yaw Sang Wan, Anh Vu Le, Chee Gen Moo, Vinu Sivanantham, Mohan Rajesh Elara
**Venue:** [IEEE Transactions on Automation Science and Engineering (T-ASE)](https://ieeexplore.ieee.org/abstract/document/10589354), vol. 22, pp. 5448–5463, 2024
**DOI:** [10.1109/TASE.2024.3421533](https://doi.org/10.1109/TASE.2024.3421533)

## Cite

```bibtex
@article{wan2024enabling,
  title   = {Enabling framework for constant complexity model in autonomous inter-reconfigurable robots},
  author  = {Wan, Ash Yaw Sang and Le, Anh Vu and Moo, Chee Gen and Sivanantham, Vinu and Elara, Mohan Rajesh},
  journal = {IEEE Transactions on Automation Science and Engineering},
  volume  = {22},
  pages   = {5448--5463},
  year    = {2024},
  publisher = {IEEE},
  doi     = {10.1109/TASE.2024.3421533}
}
```

## Funding

This research was supported by the National Robotics Programme under its National Robotics Programme (NRP) BAU, *Ermine III: Deployable Reconfigurable Robots*, Award No. **M22NBK0054**, and by A\*STAR under its *RIE2025 IAF-PP — Advanced ROS2-native Platform Technologies for Cross-sectorial Robotics Adoption* programme, Award No. **M21K1a0104**.
