'''
The controllers module is mainly responsible for managing and coordinating the decision-making process of multiple agents,
ensuring that they can work together effectively within the environment.

In basic_controller.py, a class called BasicMAC (Multi-Agent Controller) is defined.
It serves as a controller for multiple agents and is responsible for tasks such as constructing agent inputs,
selecting actions, and managing hidden states. This ensures that agents can effectively learn and perform tasks in the environment.
'''
REGISTRY = {}

from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC