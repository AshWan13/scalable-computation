# IGBNN-μ — Complete Coverage for Inter-Reconfigurable Robots

> **Algorithm:** Glasius Bioinspired Neural Network coverage planner for robots that fuse and split mid-sweep, with minigraph decomposition.
> **Complexity:** O((log n)⁻¹) — inverse logarithmic in robot count; coverage effort per robot falls as the team grows.

## Demo

![IGBNN-μ multi-robot complete coverage with inter-reconfiguration](../assets/igbnnmu_demo.gif)

## Script

[`open_igbnnmu.py`](open_igbnnmu.py) — pure Python class `IGBNN_mu`.

```python
from igbnnmu.open_igbnnmu import IGBNN_mu
planner = IGBNN_mu(...)
```

## Paper

**Title:** *Complete Coverage Path Planning by IGBNN-μ for Scalable Inter-Reconfigurable Robots*
**Authors:** Ash Yaw Sang Wan, Jiao Yang, Mohan Rajesh Elara, Anh Vu Le
**Venue:** [IEEE *Transactions on Systems, Man, and Cybernetics: Systems*](https://ieeexplore.ieee.org/document/11700393), early access, pp. 1–14, 2026
**DOI:** [10.1109/TSMC.2026.3731065](https://doi.org/10.1109/TSMC.2026.3731065)

## Cite

```bibtex
@article{wan2026complete,
  title   = {Complete Coverage Path Planning by {IGBNN}-$\mu$ for Scalable Inter-Reconfigurable Robots},
  author  = {Wan, Ash Yaw Sang and Yang, Jiao and Elara, Mohan Rajesh and Le, Anh Vu},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  pages   = {1--14},
  year    = {2026},
  publisher = {IEEE},
  doi     = {10.1109/TSMC.2026.3731065}
}
```

## Funding

This research was supported by the National Robotics Programme under its National Robotics Programme 2.0, *LEO 1.0: A New Class of Bed Making Robot*, Award No. **M25N4N2028**, and by A\*STAR under its *RIE2025 IAF-PP — Modular Reconfigurable Mobile Robots (MR)²* programme, Grant No. **M24N2a0039**.
