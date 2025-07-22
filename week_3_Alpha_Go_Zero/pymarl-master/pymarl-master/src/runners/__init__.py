'''
The purpose of the runners module is to handle environment interaction, data collection, and the training process in MARL algorithms.
Its core function is to manage the interaction between multiple agents and the environment,
collecting data such as states, actions, and rewards at each time step for subsequent training and analysis.
This module implements two types of runners: ParallelRunner and EpisodeRunner,
which are used for running parallel environments and a single environment, respectively.
'''

REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
