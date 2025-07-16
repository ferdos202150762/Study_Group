'''
main.py is the entry point of the PyMARL framework and is responsible for setting up the experiment configuration and launching the training process.
Its main functions include:

Initializing configuration and logging:
The experiment workflow is managed using the sacred library.
It sets up a logger and defines the path for saving results.
This file captures and manages experiment outputs to ensure that standard outputs and logging information are properly recorded.

Reading and merging configuration files:
It loads algorithm and environment configurations from multiple YAML files using the _get_config function,
and merges the default, environment, and algorithm settings via recursive_dict_update.

Setting random seeds:
To ensure reproducibility of experiments, main.py sets random seeds for NumPy and PyTorch.

Launching the experiment:
The experiment is started by calling ex.run_commandline(params),
which in turn calls the run() function to execute the core MARL (Multi-Agent Reinforcement Learning) training process.
'''
import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

# Create a Sacred object to manage the experiment
SETTINGS['CAPTURE_MODE'] = "fd"
# "fd" indicates that standard output (stdout) and standard error (stderr) will be captured via file descriptors.
# This means all print outputs and error messages will be written to a file instead of being shown directly in the console.
# If set to "no", stdout and stderr will be displayed directly in the console.

logger = get_logger()
# Create a logger instance. get_logger() is a custom function that returns a configured logger.
# This logger is responsible for recording important log information during the experiment.
# The logger is typically configured with log levels, formats, output destinations (e.g., console or file), and other settings.

ex = Experiment("pymarl")
# Create a Sacred Experiment object.
# The Experiment class is used to define and manage an experiment.
# "pymarl" is the name of the experiment and helps identify it.

ex.logger = logger
# Associate the previously created logger with the Sacred Experiment object.

ex.captured_out_filter = apply_backspaces_and_linefeeds
# Set the output filter.
# apply_backspaces_and_linefeeds is a utility function used to handle carriage returns and line feeds in the standard output.
# It ensures that log output remains readable and well-formatted in the presence of control characters.

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# Define the path where experiment results will be stored.


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)

@ex.main         # @ex.main is a Sacred decorator that marks this function as the main entry point of the experiment. It will be called when the experiment starts.

# my_main is the main experiment function decorated with Sacred. When the experiment starts, this function is executed.
# It sets random seeds and initiates the experiment workflow.
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)   # This line creates a deep copy of the experiment config _config and stores it in config. This ensures that modifications to config do not affect the original _config, which may be used elsewhere.
    np.random.seed(config["seed"])  # Sets the random seed for NumPy to make the experiment deterministic and reproducible. The seed is retrieved from the config.
    th.manual_seed(config["seed"])  # Similarly, sets the random seed for PyTorch.
    config['env_args']['seed'] = config["seed"]  # Sets the same random seed for environment initialization to ensure controlled randomness.

    # run the framework
    run(_run, config, _log)         # Runs the main experiment workflow

# The _get_config function is used to extract a specific config file from command-line parameters and load it as a dictionary.
def _get_config(params, arg_name, subfolder):
    config_name = None                         # Initialize config_name as None to store the config file name specified in the command-line parameters.
    for _i, _v in enumerate(params):           # Iterate over the params list to check each parameter.
        if _v.split("=")[0] == arg_name:       # Split the parameter by '=' and check if the parameter name matches arg_name.
            config_name = _v.split("=")[1]     # If it matches, extract the value after '=' as config_name (the name of the config file).
            del params[_i]                     # Delete the processed parameter to avoid redundant usage later.
            break

    if config_name is not None:                # If a valid config file name was found, proceed to read the config file.
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)     # Open the specified YAML config file located in the subfolder and parse it using the yaml library, returning the config as a dictionary.
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)  # If there’s an error reading the YAML file, raise an assertion error and print the error message.
        return config_dict

# Recursively merge two dictionaries d and u. If nested dictionaries are encountered, update them recursively.
def recursive_dict_update(d, u):
    for k, v in u.items():                          # Iterate through the key-value pairs in dictionary u; k is the key, and v is the corresponding value.
        if isinstance(v, collections.Mapping):      # Check if v is a dictionary (i.e., a mapping type). collections.Mapping is an abstract base class for dict-like types in Python.
            d[k] = recursive_dict_update(d.get(k, {}), v)  # If v is a dictionary, recursively update the corresponding key k in d. If k doesn't exist in d, use an empty dictionary {} as default.
        else:
            d[k] = v                                # If v is not a dictionary, directly assign it to d[k], either updating or adding the key-value pair.
    return d                                        # Return the merged dictionary d.


# Recursively deep copy a configuration object to ensure that each element in dictionaries or lists is independently copied and does not share references with the original.
def config_copy(config):
    if isinstance(config, dict):                               # Check if config is a dictionary.
        return {k: config_copy(v) for k, v in config.items()}  # If config is a dictionary, recursively call config_copy on each key-value pair to create a new dictionary with deep-copied values.
    elif isinstance(config, list):                             # If config is a list, recursively call config_copy on each element to create a new list.
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)                                # If config is neither a dictionary nor a list, return a deep copy of config directly.


if __name__ == '__main__':
    # Deep copy the command-line arguments list sys.argv. `params` is a copied list to prevent modifications from affecting the original arguments.
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    # env_config will override configuration items with the same name in default.yaml.
    # alg_config will override existing configuration items from the previous two (i.e., it has the highest priority).
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path)) # Add a FileStorageObserver to the Sacred experiment to save logs and data to disk.

    # Run the Sacred experiment, passing the command-line arguments `params` to start the entire experiment process.
    ex.run_commandline(params)

