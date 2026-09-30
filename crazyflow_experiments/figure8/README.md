# Figure-8 tracking
BPTT and PPO policies that track a figure-8 using the attitude interface. The trajectory duration is selected with `--preset`. Deployment is configured through `crazyflow_experiments/figure8/config.toml`. With `--dr_scale`, PPO trains under domain randomization of the mass, inertia, rotor dynamics, thrust and torque curves and external disturbances, and `--eval_dr_scale` evaluates the trained policy under randomization.

```bash
pixi run -e figure8 python crazyflow_experiments/figure8/train_bptt.py
pixi run -e figure8 python crazyflow_experiments/figure8/train_bptt.py --preset 5_5s
pixi run -e figure8 python crazyflow_experiments/figure8/train_ppo.py
pixi run -e figure8 python crazyflow_experiments/figure8/train_ppo.py --preset 5_5s
pixi run -e figure8 python crazyflow_experiments/figure8/train_ppo.py --dr_scale 1.0 --eval_dr_scale 1.0
pixi run -e figure8 python crazyflow_experiments/figure8/sim.py
pixi run -e figure8 python crazyflow_experiments/figure8/deploy.py
```

`sim.py` flies the controller of `config.toml` (`ppo`, `bptt`, `mpc` or `mellinger`) in simulation, `deploy.py` on the real Crazyflie. Start the tracker and the estimator as described in the [main readme](../../README.md) before deploying. Recordings land in `results/figure8/`.

Checkpoints are named after the algorithm, the loop time of the preset and the randomization scale,
e.g. `saves/figure8/ppo_f8_5.5s_dr0.2_model.ckpt`, built by `checkpoint_name` in `__init__.py`.
Deployment loads the policy trained for the loop time in `config.toml` and refuses to fly when it
has not been trained yet.

Training metrics are logged to W&B. Simulations and deployments save recordings under
`results/figure8/`; `utils/errors.py` defines the tracking RMSE used to evaluate them.
