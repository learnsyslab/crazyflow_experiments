"""Figure-8 tracking policies trained with BPTT and PPO.

Each `train_*.py` covers one algorithm, with the trajectory duration selected by `--preset`. The
policies and the MPC fly in simulation with `sim.py` and on the real drone with `deploy.py`. All run
in the `figure8` Pixi environment.
"""

from __future__ import annotations

from crazyflow_experiments.utils.paths import RESULTS_DIR, SAVES_DIR

MODEL_DIR = SAVES_DIR / "figure8"
RECORDINGS_DIR = RESULTS_DIR / "figure8"


def checkpoint_name(
    exp_name: str, trajectory_time: float, dr_scale: float = 0.0, seed: int | None = None
) -> str:
    """File name of a figure-8 policy checkpoint.

    The name carries everything that selects a policy: the experiment it was trained in, the loop
    time of the trajectory it was trained for and the domain randomization scale, e.g.
    `ppo_f8_5.5s_dr0.2_model.ckpt`. Trainers and the deployed controller build it here, so a policy
    trained for one loop time is never flown on another.

    Args:
        exp_name: Experiment name of the trainer, e.g. `ppo_f8`.
        trajectory_time: Loop time of the trajectory the policy was trained for in s.
        dr_scale: Domain randomization scale, 0 for the nominal policy.
        seed: Seed of the run when several seeds are trained side by side, else None.
    """
    name = f"{exp_name}_{trajectory_time:g}s"
    if dr_scale > 0:
        name += f"_dr{float(dr_scale)}"
    if seed is not None:
        name += f"_seed{seed}"
    return f"{name}_model.ckpt"
