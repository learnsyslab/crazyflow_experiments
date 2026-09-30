# System identification

For a good sim2real performance, we obviously first need to obtain a good model. In the following, we will obtain two models: The inertia **J** for a first principles based model (assuming thrust curves etc is known) and the parameters for a simplified model.

## First principles model
We rely on the following publication to identify **J**: ["Data-Driven System Identification of Quadrotors Subject to Motor Delays"](https://arxiv.org/pdf/2404.07837). We adapted their work to work directly with the wireless communication of the Crazyflie. Check out their [Github](https://github.com/arplaboratory/data-driven-system-identification).

First, go to `sysid/config.toml` and set `trajectories = "lol_sysid"`, the model of your drone and its id and channel under `[drone.cf]`, and keep `log_variables` filled. Then launch the tracker and the estimator as described in the [main readme](../../README.md).

Now we can collect data with 
```bash
pixi run -e sysid python crazyflow_experiments/sysid/collect_data.py
```

The firmware variables land in `results/sysid/lol_sysid_<model>_<date>.csv`, resampled to the control frequency. Then, change the `csv_files` list in the `lol_sysid.py` script and run it
```bash
pixi run -e sysid python crazyflow_experiments/sysid/lol_sysid.py
```

## Abstracted model
To identify the abstracted model, launch the tracker and the estimator as described in the [main readme](../../README.md).

Then, set `trajectories` in `sysid/config.toml` to either `sysid_train` or `sysid_valid` for the training or validation dataset, and empty `log_variables`. For more powerful drones, such as the Crazyflie brushless, you can add `_thrust` to the set name. The sets themselves are listed under `[trajectory_sets]` in `sysid/config_trajectories.toml` and flown one after another with a linear transition wherever one trajectory does not end at the start of the next. Now you can start to record a rosbag in a new terminal, ideally in the rosbag path:
```bash
cd results/sysid/rosbags
pixi run -e sysid ros2 bag record /tf /drones/cf<your_drone_id>/command
```

The trajectory is then launched with 
```bash
pixi run -e sysid python crazyflow_experiments/sysid/collect_data.py
```

After collecting the data, convert the recordings to a csv and fit the model. A Crazyflie is converted from the rosbag with `data_converter_mocap.py`, a PX4 vehicle from the ulog its flight controller wrote, which goes to `results/sysid/ulogs/` and is read by `data_converter_px4.py`. Both write the same columns, so the fit does not care where the flight came from. Set `bag_paths` in the converter and `FILENAME_TRAIN` and `FILENAME_VALID` at the top of `fit_simplified_model.py` to the recordings, `FILENAME_VALID = None` fits without validation data.
```bash
pixi run -e sysid python crazyflow_experiments/sysid/data_converter_mocap.py  # or data_converter_px4.py
pixi run -e sysid python crazyflow_experiments/sysid/fit_simplified_model.py --dynamics so_rpy_rotor_drag
```

`--dynamics` picks the translational model, `so_rpy` for the plain thrust, `so_rpy_rotor` with the rotor lag and `so_rpy_rotor_drag` with the body drag on top, which is the default. The identified parameters land in `results/sysid/params_<model>_<dynamics>_<date>.toml`, in the format of crazyflow's `dynamics/<dynamics>/params.toml`, so they can be pasted there.

## Thrust curve

The thrust curve of one motor is measured on a thrust stand and fitted separately, and the result goes into the drone's table in `config_drones.toml`. On a PX4 vehicle, `px4_random_thrust.py` runs on the companion computer and commands random normalized thrusts through PX4's thrust model and battery compensation, with only the motor on the stand spinning. The terminal and the display of the stand are screen recorded, `extract_thrust_from_video.py` reads the command-thrust pairs out of that recording, and `fit_thrust_to_thrust.py` fits the measured thrust over the commanded one. `fit_pwm_to_thrust.py` is the counterpart for the ramp of `PX4_thrust_sweep.sh`, which steps the motor through `actuator_test` and so measures PX4's thrust model over the actuator output, without the battery compensation.

```bash
pixi run -e sysid python crazyflow_experiments/sysid/px4_random_thrust.py
pixi run -e sysid python crazyflow_experiments/sysid/extract_thrust_from_video.py results/sysid/<recording>.webm
pixi run -e sysid python crazyflow_experiments/sysid/fit_thrust_to_thrust.py
```
