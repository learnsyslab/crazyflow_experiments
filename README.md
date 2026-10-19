# Crazyflow Experiments
<div align="center">

  [![arXiv](https://img.shields.io/badge/arXiv-2606.01478-b31b1b.svg)](https://arxiv.org/abs/2606.01478)
  [![Powered by Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
  [![Crazyflow](https://img.shields.io/badge/Powered_by-Crazyflow-3b82f6?labelColor=grey)](https://github.com/learnsyslab/crazyflow)

</div>

Experiment code for [Crazyflow](https://github.com/learnsyslab/crazyflow). It covers
simulator speed benchmarks, sim-to-real validation, system identification, learned controllers,
sampling-based obstacle avoidance, drone racing, and vision policies from Gaussian splats.

## Installation

We use pixi for the environments. Install pixi with

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

and restart your shell, then fetch the submodules with

```bash
git submodule update --init --recursive
```

One of the simulators we compare against relies on IsaacGym, which is not available from PyPI. It has to be fetched before installing anything.

```bash
pixi run --frozen setup-isaacgym
pixi install
```

Environments build their native dependencies through activation scripts in `tools/`, so the first
entry into one takes a while.

### Environments

Each experiment has its own environment to avoid dependency conflicts between different simulations and deployment tools.

| Environment | Experiments |
|---|---|
| `crazyflow` | `benchmarks/` |
| `sim2real` | `sim2real/` |
| `sysid` | `sysid/` |
| `racing` | `racing/` |
| `mppi` | `mppi_obstacles/` |
| `figure8`, `learn2fly`, `throw` | `figure8/`, `learn2fly/`, `throw/` |
| `vision` | `vision/` |
| `pybullet`, `l2f`, `flightmare`, `diff-aero`, `aerial-gym` | Baselines for `benchmarks/` and `sim2real/` |

Enter an environment with `pixi shell -e <env>`, or run a single command in it with
`pixi run -e <env> <command>`.

## Real drone deployment

Every script that flies the real drone takes its state from motion capture and talks to the drone
over the radio. Launch the tracker and the estimator in separate terminals first.

```bash
pixi run -e sim2real ros2 launch motion_capture_tracking launch.py
pixi run -e sim2real python submodules/drone-estimators/drone_estimators/ros_nodes/ros2_node.py \
    --drone_name cf<your_drone_id> --legacy
```

Any environment with ROS works. The drone id, radio channel and model come from the config file of the experiments.

### PX4 vehicles

A PX4 vehicle is flown from its companion computer through MAVROS instead. Install the repository
there with pixi as well: the `sim2real` and `sysid` environments also resolve for `linux-aarch64`,
so `pixi install -e sysid` works on a Jetson. The ROS environments build `mavros_msgs` into
`ros_ws/` on activation (`tools/setup_mavros_msgs.sh`, pin `MAVROS_MSGS_REF` to the MAVROS version
of the companion computer) and `motion_capture_tracking` (`tools/setup_mocap.sh`). On aarch64 acados
is not built, cfclient is not installed, and JAX runs on the CPU. Set the drone model in the config
to anything but a Crazyflie, e.g. `hb_x500`, and describe the vehicle in its table of
`sysid/config_drones.toml`, thrust curve and limits included, and MAVROS in `[drone.px4]`. Start the
script, then arm and flip the RC mode switch to OFFBOARD: it flies to the start of the experiment,
through the trajectory, and holds position at the end until you take over with the RC, which works
at any time. `localization` in `[drone.px4]` picks where the estimate comes
from, `"gps"` outdoors on an RTK fix or `"mocap"` indoors, and the flight aborts into PX4's landing
mode when that measurement goes stale. The scripts see the vehicle in a global frame with x north, y
west and z up, anchored at the takeoff position with `"gps"` and at the motion capture origin with
`"mocap"`, while MAVROS reports in ENU relative to the estimator origin.

Indoors the vehicle needs the motion capture pose on `/mavros/vision_pose/pose`. The Holybro launch
option of `motion_capture_tracking` publishes it there directly, without a relay node and without
the tf tree:

```bash
pixi run -e sim2real ros2 launch motion_capture_tracking holybro_launch.py
```

The motion capture type, the hostname and the `hb1` rigid body it tracks are in
`config/holybro_cfg.yaml` of the package. The launch file pins `ROS_AUTOMATIC_DISCOVERY_RANGE` to
`LOCALHOST`, so run it on the companion computer next to MAVROS, where the node reaches the motion
capture system over its own connection to `hostname`, or run it on the ground station and set the
range to `SUBNET` so MAVROS on the vehicle sees the topic.

```bash
pixi run -e sysid python crazyflow_experiments/sysid/collect_data.py
pixi run -e sim2real python crazyflow_experiments/sim2real/deploy.py
```

To check a trajectory before flying it, plot the config:

```bash
pixi run -e sim2real python crazyflow_experiments/control/trajectory_visualizer.py crazyflow_experiments/sysid/config.toml
```

## Experiments

Each experiment lives in its own folder and has its own readme with the setup and the commands.

| Experiment | Readme |
|---|---|
| System identification | [`crazyflow_experiments/sysid/`](crazyflow_experiments/sysid/README.md) |
| Sim2real | [`crazyflow_experiments/sim2real/`](crazyflow_experiments/sim2real/README.md) |
| Benchmarking | [`crazyflow_experiments/benchmarks/`](crazyflow_experiments/benchmarks/README.md) |
| Racing | [`crazyflow_experiments/racing/`](crazyflow_experiments/racing/README.md) |
| Figure-8 tracking | [`crazyflow_experiments/figure8/`](crazyflow_experiments/figure8/README.md) |
| Reach position | [`crazyflow_experiments/learn2fly/`](crazyflow_experiments/learn2fly/README.md) |
| Throw and stabilize | [`crazyflow_experiments/throw/`](crazyflow_experiments/throw/README.md) |
| MPPI obstacle avoidance | [`crazyflow_experiments/mppi_obstacles/`](crazyflow_experiments/mppi_obstacles/README.md) |
| Vision | [`crazyflow_experiments/vision/`](crazyflow_experiments/vision/README.md) |

For the learned controllers (figure-8 tracking, reach position, throw and stabilize), environments and agents are implemented in `core/`. Checkpoints and recordings go under `saves/` and `results/`.

## Development
Since our experiments cover multiple environments, the tests also need to run in multiple environments. Each test module therefore declares the environment it needs and skips itself elsewhere. This allows us to run a single test-all command that checks everything

```bash
pixi run test       # runs the suite in the active environment
pixi run test-all   # runs all tests in their respective environments
```

Results go to `results/` and checkpoints to `saves/`, both resolved through `utils/paths.py`.

## Citation

If you use this code, please cite the Crazyflow paper.

```bibtex
@misc{schuck2026crazyflow,
      title={Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX},
      author={Martin Schuck and Marcel P. Rath and Yufei Hua and Abhishek Goudar and SiQi Zhou and Angela P. Schoellig},
      year={2026},
      eprint={2606.01478},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.01478},
}
```
