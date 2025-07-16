from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1 #batch_size must be equal to 1 to ensure that only one environment is processed during each training or testing step

        # Get the environment object from the env_REGISTRY registry.
        # env_REGISTRY[self.args.env] is an environment class (or constructor),
        # and **self.args.env_args unpacks and passes the arguments to the environment constructor.
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = self.env.episode_limit # Get the episode limit of the environment
        self.t = 0 # Initialize to 0, used to track the time step of the current episode

        self.t_env = 0 # Initialize to 0, used to track the time step of the current environment

        # Used to store the returns for training and testing, respectively
        self.train_returns = []
        self.test_returns = []
        # Used to store statistics for training and testing, respectively
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        # Time step used to record the last training statistics, initialized to a very small value to ensure recording during the first training.
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac): # Set up batch generation and multi-agent control related settings
        # Use the partial function to create a partially applied EpisodeBatch function.
        # EpisodeBatch is used to create new batch data, and partial presets some of the function's parameters,
        # while the remaining parameters (scheme, groups, preprocess, device) are provided when the function is called.
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self): # Return environment information
        return self.env.get_env_info()

    def save_replay(self): # Store replay information
        self.env.save_replay()

    def close_env(self): # Close the environment
        self.env.close()

    def reset(self): # Reset the state of the class
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False # Flag variable indicating whether the current episode has ended
        episode_return = 0 # Accumulate the total return of the current episode
        self.mac.init_hidden(batch_size=self.batch_size) # Initialize the hidden state of the multi-agent controller (MAC)

        while not terminated: # Loop body, continues until the episode ends, i.e., stops when terminated is True

            # Collect environment state, available actions, and observation data at the current time step
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            # Update data by adding pre_transition_data to the data at the current time step self.t
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            # Use the multi-agent controller (MAC) to select actions at the current time step.
            # The select_actions method selects actions from the current batch data,
            # and the test_mode parameter determines whether it is in test mode.
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # Execute one step in the environment; actions[0] is the action of the first agent.
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            # Collect data after executing actions at the current time step
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            # Update the batch data by adding post_transition_data to the data at the current time step self.t
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1 # Increment the value of the time step

        # At the end of the episode, collect the final environment state, available actions, and observation data.
        # Update the batch data by adding last_data to the data at the current time step self.t
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # At the end of the episode, select actions based on the final state and update the batch data
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        # Determine the current statistics and return lists based on test_mode, and set the log prefix
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        # Update the statistics cur_stats, including all keys from environment information,
        # accumulate current episode information, increment episode count by 1,
        # and add the current time step self.t to the episode length (ep_length)
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # If not in test mode, increment the environment time step self.t_env
        if not test_mode:
            self.t_env += self.t

        # Add the return value of the current episode, episode_return, to the return list cur_returns
        cur_returns.append(episode_return)

        # If in test mode and the number of test returns reaches the set amount self.args.test_nepisode, call the _log function to record data.
        # If in training mode, and the difference between the current time step self.t_env and the last logged time step self.log_train_stats_t
        # reaches the set logging interval self.args.runner_log_interval, also call the _log function to record data.
        # If self.mac.action_selector has an epsilon attribute, record the value of epsilon.
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        # Log the mean and standard deviation of the returns
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear() # Clear the list of returns

        # Log the mean of the statistics (excluding n_episodes)
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear() # Clear the statistics dictionary

