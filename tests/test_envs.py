from tests.conftest import module_requires_env

module_requires_env("vision")

from crazyflow_experiments.core.envs.vision_reach_env import VisionReachEnv  # noqa: E402
from crazyflow_experiments.vision.scenes import splat  # noqa: E402


def test_vision_reach_env():
    env = VisionReachEnv.create(scene=splat("lab", "lab_30k"))
    env = env.reset(env)[0]
    env.step(env, env.action_space.sample())
