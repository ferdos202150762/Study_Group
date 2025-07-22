'''
Classes in the Learners module, such as QLearner and COMALearner,
are responsible for executing the core training logic of reinforcement learning algorithms.

They use experience data to update the agent's policy and value functions.
The Learners module is a crucial component of the PyMARL framework,
as it transforms experience data into signals that guide policy and value function updates,
ultimately determining how the agent learns from the environment and optimizes its decision-making.
'''

from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
