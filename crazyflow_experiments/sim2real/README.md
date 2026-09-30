# Sim2real

We fly the same trajectory in every simulator and on the real drone. To run the simulation experiments, change the setup in `crazyflow_experiments/sim2real/config.toml`. Then run the simulation with
```bash
pixi run -e sim2real python crazyflow_experiments/sim2real/sim_crazyflow.py
```

For the other simulators, use their own environment. CrazySim runs in a docker container instead, see below.
```bash
pixi run -e pybullet python crazyflow_experiments/sim2real/sim_pybullet.py
pixi run -e l2f python crazyflow_experiments/sim2real/sim_l2f.py
pixi run -e flightmare python crazyflow_experiments/sim2real/sim_flightmare.py
```

l2f and Flightmare only expose rotor commands in their python API, so both scripts drive them with crazyflow's controllers.

Every simulation script repeats its rollout `n_rollouts` times (see `[sim]` in `config.toml`) and numbers the recordings like the deployments, so that `compute_rmses.py` can pair the k-th rollout with the k-th real flight.
We also added a script that automates multiple simulations in one run. For that, add all the settings to `utils.py` and run
```bash
pixi run -e sim2real python crazyflow_experiments/sim2real/sim_automation.py
```

To run the deployment, start the tracker and the estimator as described in the [main readme](../../README.md) and then run
```bash
pixi run -e sim2real python crazyflow_experiments/sim2real/deploy.py
```

The recordings are stored in `results/sim2real/<model>/`, one folder per drone model. Recordings are timed in trajectory time, zero where the trajectory starts and the soft start before it. Afterwards, compute the RMSEs of every simulator against the real flights and test them against each other. `compute_rmses.py` computes the RMSE over the middle run of the trajectory, pairing rollout k with real flight k, and writes `<drone>_RMSEs.toml` into the folder of the model, which `compute_stats.py` reads for the paired one-sided t-tests and the intersection-union test over the trajectories. The simulators that enter the comparison are listed in `compute_rmses.SIMULATORS`.
```bash
pixi run -e sim2real python crazyflow_experiments/sim2real/compute_rmses.py
pixi run -e sim2real python crazyflow_experiments/sim2real/compute_stats.py
```

## Gym Pybullet Drones
`sim_pybullet.py` overrides the stock `cf2x.urdf` mass and inertia at runtime with the identified parameters of our drone from `crazyflow.drones`, in the PyBullet body as well as in the constants `BaseAviary` derives from them. The physics name selects the attitude loop and the drag model, see `sim_pybullet.VARIANTS`:
`pybullet` is the stock setting (DSLPIDControl, no drag), the `-pybdrag`/`-drag` suffixes add gym-pybullet-drones' own or crazyflow's identified drag, and the `-mellinger` infix swaps in crazyflow's Mellinger attitude loop. `-rotor` puts crazyflow's rotor dynamics in front of the attitude loop, and `-fwclip` widens the torque clip of DSLPIDControl from gym-pybullet-drones' ±3200 to the firmware's ±32000 PWM, which keeps the ringing of the lagged loop from turning into a flip.

## CrazySim
This is the only simulator we are not able to run with our pixi environment, since cfclient commands can't reach the sim. Instead, we use a docker container for the Simulator. The Dockerfile lives at `crazyflow_experiments/sim2real/CrazySimDocker`. The mass and the inertia are always patched to our drone setup during the build; all other patches are opt-in build args, see the header of that file:

```bash
sudo docker build -f crazyflow_experiments/sim2real/CrazySimDocker -t crazysim:base .
sudo docker build -f crazyflow_experiments/sim2real/CrazySimDocker --build-arg GZ_FIX=1 -t crazysim:gz-fix .
sudo docker build -f crazyflow_experiments/sim2real/CrazySimDocker --build-arg MJ_FIX=1 -t crazysim:mj-fix .
sudo docker build -f crazyflow_experiments/sim2real/CrazySimDocker --build-arg SIM_TICK=1 -t crazysim:simtick .
```

`GZ_FIX` corrects the rolling moment coefficient of the fourth rotor and makes the thrust chain of the Gazebo model consistent, `MJ_FIX` gives the MuJoCo backend a fast rotor time constant, and `SIM_TICK` makes the firmware count its FreeRTOS ticks from the simulator instead of the wall clock, which takes the real-time factor out of the firmware's estimator.

To run the sim2real experiments for one simulation variant, headless and fully automated (recordings land in `results/sim2real/cf2x_L250/*_<name>_NN.pkl`):

```bash
sudo docker run --rm -v <path/to/crazyflow_experiments>:/workspace:rw \
    -v <path/to/crazyflow>:/crazyflow:ro crazysim:base \
    python3 /workspace/crazyflow_experiments/sim2real/run_crazysim.py crazysim-gz
```

with `crazysim-gz` and `crazysim-mj` on `crazysim:base`, `crazysim-gz-fix` on `crazysim:gz-fix`, and `crazysim-mj-fix` on `crazysim:mj-fix`.

The physics name also selects the model changes the script itself makes to the MuJoCo backend, see `load_mj_backend`: `crazysim-mj-reaction` adds the rotor reaction torque the backend omits, `crazysim-mj-reaction-rotor` swaps in crazyflow's identified rotor model on top, `origpacing` paces the physics like CrazySim's own loop instead of in real time, and `simtick` tells the script that the firmware of the image counts simulator ticks, so the control loop runs on simulation time as well and needs the `crazysim:simtick` image.

For interactive use with the Gazebo GUI instead, allow X access with `xhost +local:docker`, then launch the container and start the sim:
```bash
sudo docker run -it --rm \
    --name crazysim \
    --net=host \
    --gpus all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env="DISPLAY=$DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="<path/to/crazyflow_experiments>:/workspace:rw" \
    crazysim:base
bash tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_singleagent.sh -m crazyflie -x 0 -y 0
```

Now we open another terminal and exec into the container with `sudo docker exec -it crazysim bash`. The experiments need to be installed too:
```bash
cd /workspace/
pip install -e .
```

The experiment can then be launched with
```bash
python /workspace/crazyflow_experiments/sim2real/sim_crazysim.py
```

These commands run inside the container, which has no pixi, so they call `python` directly.
