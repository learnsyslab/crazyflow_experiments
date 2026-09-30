# MPPI obstacle avoidance
A multi-modal MPPI controller flies the drone through obstacles.

```bash
pixi run -e mppi python crazyflow_experiments/mppi_obstacles/sim.py
pixi run -e mppi python crazyflow_experiments/mppi_obstacles/deploy.py
```

The MPPI parameters, the cost weights and the obstacle layout are in `crazyflow_experiments/mppi_obstacles/config.toml`. The real pillars are tracked by motion capture, and `generate_markers.py` generates marker layouts that the tracker can tell apart.

Use `--record True` with `sim.py` to save trajectories, references and planner rollouts for analysis.
