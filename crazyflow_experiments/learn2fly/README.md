# Reach position
We mimick the setup of [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081) and train TD3 policies that command rotor velocities directly. `export.py` writes the trained actor as a C header so the policy can run onboard.

```bash
pixi run -e learn2fly python crazyflow_experiments/learn2fly/train.py
pixi run -e learn2fly python crazyflow_experiments/learn2fly/export.py
```

`sim.py` flies the trained policy along the trajectory of `config.toml` in the Crazyflow simulator, where the policy chases the current point of the trajectory. `deploy.py` flies the exported policy on the real Crazyflie: the firmware runs the actor as its controller 8 and takes the reference state streamed over the radio as its goal. Start the tracker and the estimator as described in the [main readme](../../README.md) first. Both store their recordings in `results/learn2fly/`.

```bash
pixi run -e learn2fly python crazyflow_experiments/learn2fly/sim.py
pixi run -e learn2fly python crazyflow_experiments/learn2fly/deploy.py
```
