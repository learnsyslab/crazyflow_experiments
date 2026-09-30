# Benchmarking
In this experiment, we benchmark the performances of multiple simulators in terms of simulation speed. Each benchmark saves its timings to `results/benchmark`.

Dynamics benchmarks:
```bash
pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_dynamics.py && \
pixi run -e diff-aero python crazyflow_experiments/benchmarks/diff_aero_dynamics.py && \
pixi run -e aerial-gym python crazyflow_experiments/benchmarks/aerial_gym_dynamics.py && \
pixi run -e pybullet python crazyflow_experiments/benchmarks/gym_pybullet_drones_dynamics.py && \
pixi run -e l2f python crazyflow_experiments/benchmarks/l2f_dynamics.py && \
pixi run -e flightmare python crazyflow_experiments/benchmarks/flightmare_dynamics.py
```

Gradient benchmarks:
```bash
pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_gradients.py && \
pixi run -e diff-aero python crazyflow_experiments/benchmarks/diff_aero_gradients.py
```

Rendering benchmarks:
```bash
pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_render.py && \
pixi run -e diff-aero python crazyflow_experiments/benchmarks/diff_aero_render.py && \
pixi run -e flightmare python crazyflow_experiments/benchmarks/flightmare_render.py && \
pixi run -e vision python crazyflow_experiments/benchmarks/crazyflow_splat_render.py
```

Flightmare and l2f only expose rotor level commands. Simulators that offer both interfaces take `--control rotor_vel` to be benchmarked under the same settings. Each interface writes its own parquet.
```bash
pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_dynamics.py --control rotor_vel && \
pixi run -e aerial-gym python crazyflow_experiments/benchmarks/aerial_gym_dynamics.py --control rotor_vel && \
pixi run -e pybullet python crazyflow_experiments/benchmarks/gym_pybullet_drones_dynamics.py --control rotor_vel
```

`crazyflow_scaling.py` measures how crazyflow scales over the accelerators of one node. It targets a multi-accelerator machine.
```bash
pixi run -e crazyflow python crazyflow_experiments/benchmarks/crazyflow_scaling.py
```

Once the timings are collected, `stats.py` reports throughput comparisons and statistical tests.
```bash
pixi run -e crazyflow python crazyflow_experiments/benchmarks/stats.py
```
