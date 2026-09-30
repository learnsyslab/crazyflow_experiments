# Racing
In this experiment, we run the code from the [LSY Drone Racing Challenge](https://github.com/utiasDSL/lsy_drone_racing) with a model predictive contouring controller. The track and the controller are defined in `crazyflow_experiments/racing/config/track.toml`.

We can directly run the simulation with
```bash
pixi run -e racing python crazyflow_experiments/racing/sim.py --render True
```

To deploy, launch the tracker and the estimator as described in the [main readme](../../README.md), and then use
```bash
pixi run -e racing python crazyflow_experiments/racing/deploy.py
```
