# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Experiment codebase for [Crazyflow](https://github.com/learnsyslab/crazyflow). It covers simulator speed benchmarks, sim-to-real validation, system identification, learned controllers, sampling-based obstacle avoidance, drone racing, and vision policies from Gaussian splats.

## Environment Setup

The project uses **Pixi** for environment management (conda-based). No single environment can hold
every dependency, so each experiment domain has its own environment, defined in `pyproject.toml`.

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

Most scripts document the environment they need in their module docstring, in the form
`pixi run -e <env> python <path/to/script.py>`. The environments built on the `kilted` feature run
`tools/setup_mocap.sh` and `tools/setup_acados.sh` on shell entry.

## Common Commands

### Tests

```bash
pixi run test       # runs the suite in the active environment
pixi run test-all   # run all tests in their respective environments
```

Each test module declares the environment it needs through `tests/conftest.py` and skips itself
elsewhere, so the same suite covers every environment without failing for the wrong one.

### Running Experiments

Training, simulation, deployment and evaluation scripts all follow the pattern from their module
docstring, so only the cases that deviate are listed here.

- `sim2real/sim_pybullet.py` runs in the `pybullet` environment rather than in `sim2real`, and
  `sim2real/sim_crazysim.py` runs inside the container, not under pixi.
- `sim2real/deploy.py` and `sysid/collect_data.py` share their flight through `deploy/experiment.py`:
  a Crazyflie flies over the radio, any other drone model is a PX4 vehicle flown through MAVROS on
  its companion computer via `deploy/px4.py`. The mass, pwm range and thrust curve of every model live
  in `sysid/config_drones.toml`, the Crazyflie radio settings under `[drone.cf]`,
  the MAVROS settings of the PX4 vehicle under `[drone.px4]` of the config. The
  pilot starts the script, then arms and switches to OFFBOARD, from where the script flies to the
  start and holds position at the end until the pilot takes over and lands. `localization` in
  `[drone.px4]` picks the estimate, `"gps"` on an RTK fix or `"mocap"`, and a stale measurement
  aborts the flight into PX4's landing mode.
- `control/trajectory_visualizer.py <config.toml>` plots the trajectory a config describes, a single
  curve from `trajectory` or a set from `trajectories` in its experiment section.
- `sim2real/sim_automation.py` sweeps the grid of models, physics backends and trajectory scalings
  defined in `sim2real/utils.py` instead of reading `config.toml` once.
- Every sim2real simulation script repeats its rollout `n_rollouts` times (`[sim]` in `config.toml`)
  and numbers the recordings like the deployments, `<experiment>_<physics>_NN.pkl`, in
  `results/sim2real/<model>/`, one folder per drone model that also holds its RMSE and stats tables.
  Recordings are timed in trajectory time, zero where the trajectory starts and the soft start
  before it. `sim2real/compute_rmses.py` pairs rollout k with real flight k over the middle run of the trajectory, and `sim2real/compute_stats.py`
  runs the paired one-sided t-tests and the intersection-union test on those RMSEs.
- The sysid scripts form a pipeline. `collect_data.py` flies the trajectory set that `config.toml`
  names among the `[trajectory_sets]` of `config_trajectories.toml` while a rosbag records, and stores the firmware variables of
  `log_variables`, resampled to the control frequency, as csv in `results/sysid/` for the low-level
  identification. `data_converter_mocap.py` turns the Crazyflie bags into csv;
  `data_converter_px4.py` does the same from the PX4 ulog in `results/sysid/ulogs/` (needs
  `vehicle_local_position` at full rate via the logger topics file), finds the trajectory as the
  offboard stretch flown with attitude setpoints, widened half a second on both sides because
  `vehicle_control_mode` is logged at 2 Hz and the flight hovers on both sides, and cuts estimator
  resets and losses of the measurements that hurt the estimate into separate csvs. It reads the
  localization off the estimator's fusion flags rather than a flag: with `"gps"` the origin is the
  home position and RTK fix losses are cut, with `"mocap"` the origin is the estimator origin, which
  is the motion capture origin, and dropouts of the motion capture pose are cut instead. Then
  `fit_simplified_model.py`
  or `lol_sysid.py` fits the model. Their input paths are constants at the top of the file rather
  than CLI arguments. `fit_simplified_model.py` writes the identified table to
  `results/sysid/params_<model>_<dynamics>_<date>.toml` in the format of crazyflow's
  `dynamics/<dynamics>/params.toml`, so it can be pasted there.
- `sysid/extract_thrust_from_video.py` reads a thrust-stand screen recording in `results/sysid/`.
  The user marks the terminal pane and the display digits, the script detects every printed
  `d.dd` command line, reads its digits (grouped by appearance, the user labels each digit once),
  decodes the seven-segment display before the next command and writes `<video>.csv` after a
  paged review step for correcting unreadable values.
- `throw/collect.py` replays the recorded throws through the training checkpoints and writes
  success curves for evaluating recovery rate.
- `learn2fly/export.py` writes the trained actor as a C header for the onboard firmware.

**Real drone deployment** requires launching in separate terminals first:
```bash
ros2 launch motion_capture_tracking launch.py
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf<drone_id> --legacy
```

A PX4 vehicle takes its pose from MAVROS instead, which the motion capture node feeds directly
through the Holybro launch option of `motion_capture_tracking`:
```bash
pixi run -e sim2real ros2 launch motion_capture_tracking holybro_launch.py
```
It publishes the pose of the `hb1` rigid body straight to `/mavros/vision_pose/pose`, so no relay
node stands between the motion capture system and MAVROS, and leaves the tf tree off, which MAVROS
does not read. The motion capture type, the hostname and the body to topic mapping are
`config/holybro_cfg.yaml` of the package, and the plain `launch.py` still comes up on the tf tree
alone for the Crazyflies. Both launch files pin `ROS_AUTOMATIC_DISCOVERY_RANGE` to `LOCALHOST`, so
the poses stay on the machine that runs the node. The companion computer builds the package as well
(`linux-aarch64` runs `setup_mocap.sh` and `setup_mavros_msgs.sh`, no acados), so the node can run
right next to MAVROS; from a ground station it reaches that MAVROS only with `SUBNET`.

## Architecture

### Module Structure

```
crazyflow_experiments/
├── control/         # Trajectories, controllers, config dataclasses, takeoff and loop timing
├── deploy/          # Radio link to the real Crazyflie and its takeoff routine
├── core/            # Shared RL environments and agents
├── utils/           # Shared paths, ROS helpers and position errors
├── benchmarks/      # Simulator speed benchmarking
├── sim2real/        # Sim-to-real validation pipeline
├── sysid/           # System identification
├── figure8/         # Figure-8 tracking policies (BPTT, PPO)
├── learn2fly/       # Reach-position policies (TD3)
├── throw/           # Throw-and-stabilize policies (BPTT)
├── mppi_obstacles/  # MPPI obstacle avoidance
├── racing/          # Trajectory planning & drone racing
└── vision/          # Vision training, simulation, deployment and scene reconstruction
```

### Shared control and deployment

`control/` holds everything the trajectory-flying experiments have in common. `trajectories.py`
defines the periodic curves registered in `TRAJECTORIES` and `CompositeTrajectory`, which plays
several of them one after another with linear transitions where they lie apart. `config.py`
parses the drone, controller and experiment sections that every config file shares, builds single
trajectories and trajectory sets from them, and maps a controller type to the Crazyflow control
interface it commands. The controllers all implement `controller.py`: `mellinger_controller.py` and
`model_predictive_controller.py` command attitude and collective thrust, `state_controller.py`
plays the trajectory through the state interface. `takeoff.py` builds the ramp to the hover pose,
`loop.py` keeps the control loop on frequency, and `trajectory_visualizer.py` plots a config's
trajectory.

`deploy/` wraps the drones. `crazyflie.py` owns the cflib connection and the ROS state feed,
`deploy/takeoff.py` flies the `control/takeoff.py` trajectory on hardware. `px4.py` is the same
interface for a PX4 vehicle through MAVROS: it streams position or attitude setpoints, converts the
thrust with the measured thrust curve, and reads the onboard estimate, all in a global frame anchored
at the takeoff position with `localization = "gps"` or at the motion capture origin with `"mocap"`,
x north, y west and z up. It watches whichever measurements the estimate rests on, the RTK fix or the
motion capture pose, and lands when they go stale. `experiment.py` runs a flight on either drone:
connect, start take off, track a controller, finish. The takeoff ends in a hover at the start pose, and
`track` records the flight in trajectory time, zero where the trajectory starts. The trajectory sets of `sysid/collect_data.py` end with a zero-size trajectory, a final hover.
The simulation counterpart is `control/simulator.py`, which builds the same observation dict from
the sim state, so controllers see identical inputs in simulation and on the drone.

### Key Dependencies

- `crazyflow`: Core JAX differentiable simulator.
  Also provides the drone dynamics (`crazyflow.dynamics`), onboard controllers
  (`crazyflow.control`) and hardware parameters (`crazyflow.drones`)
- `lsy_drone_racing`: Racing environment & utilities, pinned to an exact commit from GitHub
- `submodules/drone-estimators`: State estimation for real drones
- `acados`: Cloned and built by `tools/setup_acados.sh` for the MPC controllers

The remaining submodules are benchmark baselines, installed as editable packages within their
respective Pixi environments.

### RL Experiments

The three RL experiments share their building blocks through `core/`. The gymnasium tasks in
`core/envs/` all build on the common `drone_env.py` base, and `core/envs/wrappers.py` supplies
action normalization, observation flattening and action penalties. The Flax policies in
`core/agents/` cover BPTT in a stochastic and a deterministic variant, PPO and TD3.

Each experiment owns its training and deployment code and keeps its checkpoints and
recordings under `saves/<experiment>/` and `results/<experiment>/`, resolved through the constants
in `utils/paths.py` so that no module builds an output path of its own. `figure8/` trains figure-8
tracking with one `train_*.py` per algorithm, `learn2fly/` trains reach-position policies through
`train.py` and exports the actor as a C header with `export.py`, and `throw/` trains
throw-and-stabilize policies through `train.py`.

Experiments that run on hardware follow the same convention, where `controller.py` wraps the trained
policy, `deploy.py` flies it and `sim.py` runs it in simulation.

### Errors

`utils/errors.py` is the one place that defines position errors: `tracking_error` and
`tracking_rmse` of a flight against its trajectory, `sim2real_rmse` between a simulated flight,
interpolated onto the real flight's timestamps, and the real one, and `run_mask` to select the
tracked runs of a periodic trajectory.
Evaluation scripts and tests that report such an error call these instead of computing their own.

### Configuration Files

| File | Purpose |
|---|---|
| `crazyflow_experiments/sim2real/config.toml` | Drone model in `[drone]`, radio settings in `[drone.cf]`, MAVROS settings in `[drone.px4]`, physics backend, controller type, trajectory, experiment params |
| `crazyflow_experiments/figure8/config.toml` | Crazyflie, controller (`ppo`, `bptt` or `mpc`) and trajectory of the figure-8 flights |
| `crazyflow_experiments/learn2fly/config.toml` | Crazyflie and trajectory of the reach-position flights, in simulation and with the onboard TD3 policy |
| `crazyflow_experiments/throw/config.toml` | Drone and control frequency for the throw experiment |
| `crazyflow_experiments/mppi_obstacles/config.toml` | MPPI parameters, cost weights and obstacle layout |
| `crazyflow_experiments/vision/config.toml` | Drone and manual control interface for camera data collection |
| `crazyflow_experiments/sysid/config.toml` | Drone as in the sim2real config, controller, origin and trajectory set of the identification flight |
| `crazyflow_experiments/sysid/config_drones.toml` | Mass, pwm range, thrust curve and thrust limits of every drone model, one table per model, read by every experiment through `control.config.drone_params` |
| `crazyflow_experiments/sysid/config_trajectories.toml` | The `[trajectory_sets]` of the identification flights |
| `crazyflow_experiments/racing/config/track.toml` | Controller file, track layout, deployment checks |

## Code Style

- Type hints required on all functions
- No `try/except`. Fail early with explicit exceptions
- Use `pathlib` over `os`
- Use f-strings for string formatting
- Google-style docstrings
- Line length: 100 characters
- Ruff enforces: `E4`, `E7`, `E9`, `F`, `I`, `D`, `TCH`, `ANN`, `T20` (except `ANN401`)
- Array API is enabled for scipy, so Rotations and Transforms work with JAX arrays

## Special Setup Notes

**IsaacGym**: Not available from PyPI, and pixi resolves its path while solving the `aerial-gym`
environment. Run `pixi run --frozen setup-isaacgym` before `pixi install`.

**Activation builds**: The first entry into an environment builds its native dependencies, which
takes a while. `setup_mocap.sh` builds `motion_capture_tracking` into `ros_ws/`,
`setup_acados.sh` builds acados into `acados/`, `setup_mavros_msgs.sh` builds `mavros_msgs` into
`ros_ws/` (robostack has no package for it).

**Patched submodule**: Activation patches the build files of `flightmare` and compiles flightlib in
place. The submodule is marked `ignore = dirty`, so its working tree is always modified. The patch
is applied through the `git apply --reverse --check` idiom, so activation is idempotent and fails
loudly once the pinned upstream moves out from under the patch.

**CrazySim**: The only simulator that cannot be reached from a Pixi environment. It runs in the
container built from `sim2real/CrazySimDocker`, see the README.

**CUDA**: `vision` requires CUDA 12, and `aerial-gym`, `diff-aero` and the RL
environments require CUDA 12.1.

**Outputs**: Diagnostic plots go to `plots/`, results to `results/`, model saves to `saves/`, all resolved
through `utils/paths.py`.
