"""Reach-position policies trained with TD3.

We try to mimic the setup from https://arxiv.org/abs/2311.13081 in crazyflow for a fair comparison.
`train.py` trains the policy, `sim.py` flies it in simulation, `export.py` writes it as a C header
for the firmware and `deploy.py` flies that onboard policy on the real drone. Runs in the `learn2fly`
Pixi environment.
"""

from crazyflow_experiments.utils.paths import RESULTS_DIR, SAVES_DIR

MODEL_DIR = SAVES_DIR / "learn2fly"
RECORDINGS_DIR = RESULTS_DIR / "learn2fly"
