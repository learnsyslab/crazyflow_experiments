# Throw and stabilize
We train a BPTT policy to recover from a throw. We observe the initial state, predict where the drone will be once training finishes, and then train while the drone is in the air.

```bash
pixi run -e throw python crazyflow_experiments/throw/train.py
pixi run -e throw python crazyflow_experiments/throw/sim.py
pixi run -e throw python crazyflow_experiments/throw/deploy.py
```

The policy runs onboard, which needs our [firmware fork](https://github.com/learnsyslab/crazyflie-firmware) on the `crazyflow_experiments` branch. It adds a `Throw` controller whose network weights are exposed as float16 parameters in the `ctrlThrow` group. Flash it once before deploying.

```bash
git clone --recursive -b crazyflow_experiments https://github.com/learnsyslab/crazyflie-firmware && cd crazyflie-firmware
make cf21bl_defconfig  # cf2_defconfig for the Crazyflie 2.x
make -j$(nproc)
make cload
```

`deploy.py` uploads the trained weights over the radio and switches the drone to the throw controller mid-air, so the firmware doesn't have to be rebuilt for a new policy.

`collect.py` replays the training of every recorded throw with 10 seeds (42 to 51,
`--n_seeds` to change) through the training checkpoints and writes success curves.
Each curve records the recovery rate over all throws for every seed and checkpoint.

```bash
pixi run -e throw python crazyflow_experiments/throw/collect.py
```
