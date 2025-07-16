'''
The envs module in the PyMARL project provides a standard gym-style interface for defining multi-agent environments.
The MultiAgentEnv class within the envs module offers a basic interface for agent-environment interactions in multi-agent systems,
facilitating integration and extension across different environments and MARL algorithms.
'''
from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
