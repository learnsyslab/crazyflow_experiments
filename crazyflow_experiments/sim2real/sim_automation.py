"""This file runs all simulations for sim2real experiments.

Run with
    pixi run -e <env> python crazyflow_experiments/sim2real/sim_automation.py [--models ...] [--physics ...]
"""

import argparse
from pathlib import Path

import toml
from tqdm import tqdm

from crazyflow_experiments.sim2real import utils


def main(models: list[str], physics: list[str]):
    """Run the grid of all simulations for sim2real experiments.

    Args:
        models: Drone models to simulate.
        physics: Physics backends to simulate with.
    """
    # Load config
    base_config = toml.load(Path(__file__).parent / "config.toml")
    base_config["controller"]["type"] = utils.controller_type
    grid = utils.param_grid(models, physics)

    # Iterate with tqdm for progress
    for model, phys, traj, scaling, plane, speed in tqdm(grid, desc="Running simulations"):
        # Make a deep copy so we don't mutate the base config
        config = base_config.copy()

        # Modify config
        config["drone"]["model"] = model
        config["experiment"]["origin"] = utils.origins.get(
            model, base_config["experiment"]["origin"]
        )
        config["experiment"]["trajectory"] = traj
        config["experiment"]["scaling"] = scaling
        config["experiment"]["time"] = speed
        config["experiment"]["plane"] = plane
        config["sim"]["physics"] = phys
        config["sim"]["render"] = False

        # Run simulation
        match phys:
            case "first_principles":
                from crazyflow_experiments.sim2real.sim_crazyflow import main as sim_crazyflow

                sim_crazyflow(config)
            case "so_rpy":
                from crazyflow_experiments.sim2real.sim_crazyflow import main as sim_crazyflow

                sim_crazyflow(config)
            case "so_rpy_rotor":
                from crazyflow_experiments.sim2real.sim_crazyflow import main as sim_crazyflow

                sim_crazyflow(config)
            case "so_rpy_rotor_drag":
                from crazyflow_experiments.sim2real.sim_crazyflow import main as sim_crazyflow

                sim_crazyflow(config)
            case str() if phys.startswith("pybullet"):
                from crazyflow_experiments.sim2real.sim_pybullet import main as sim_pybullet

                sim_pybullet(config)
            case "l2f":
                from crazyflow_experiments.sim2real.sim_l2f import main as sim_l2f

                sim_l2f(config)
            case "flightmare":
                from crazyflow_experiments.sim2real.sim_flightmare import main as sim_flightmare

                sim_flightmare(config)
            case _:
                raise ValueError(f"Physics {phys} not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=utils.models, help="drone models")
    parser.add_argument("--physics", nargs="+", default=utils.physics, help="physics backends")
    args = parser.parse_args()
    main(args.models, args.physics)
