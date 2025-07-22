from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs

        # Create multiple communication pipes using Pipe(), each with two endpoints for communication between the parent (main) process and child (worker) processes.
        # self.parent_conns is the list of parent process endpoints
        # self.worker_conns is the list of worker process endpoints
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])

        # Get the environment creation function from the environment registry env_REGISTRY[self.args.env]
        env_fn = env_REGISTRY[self.args.env]

        # Create multiple subprocesses, each running the env_worker function to manage a single environment instance.
        # CloudpickleWrapper wraps the environment function to ensure it is serializable,
        # and partial is used to pass the necessary environment parameters.
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                            for worker_conn in self.worker_conns]

        # Loop to start each subprocess. p.daemon=True means these subprocesses will automatically terminate when the main process ends.
        for p in self.ps:
            p.daemon = True
            p.start()

        # Send a message to the first subprocess requesting environment information ("get_env_info").
        # Receive environment information from the subprocess as self.env_info, and extract the episode limit as self.episode_limit.
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        # The initialization of the following parameters is exactly the same as in episode_runner.py
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac): # The setup function is analogous to that in episode_runner.py
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns: # Send a "reset" message to all parent process connections to reset the environment in each child process
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        # Receive reset environment data from each parent process connection and store the data in pre_transition_data
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size) # Initialize the hidden states of the agents

        # terminated indicates whether each environment has terminated, envs_not_terminated is the list of indices for environments still running,
        # and final_env_infos stores additional information when each environment terminates
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # Select actions at the current time step using the agent controller, and send these actions to the corresponding environments
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            # Convert actions from GPU tensors to CPU numpy arrays for easier processing or sending to the environment.
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            # Store the agents' actions at the current time step in the batch data
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            # Send actions for each environment through the parent_conn pipes.
            # Only send actions to environments that have not terminated.
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            # Exit the main loop if all environments have terminated
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    # Each non-terminated environment receives reward, termination status, state, etc. Update cumulative returns (episode_returns) and episode lengths (episode_lengths)
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    # If an environment has terminated, add termination information to final_env_infos and update the terminated list.
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Store state, available actions, and observation information for the next time step to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        # Receive environment statistics from each subprocess via their parent connections,
        # and collect them into the env_stats list.
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # Determine the current statistics; based on test_mode, decide whether it is training or testing mode, and select the corresponding statistics and return storage.
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        # Update statistics by merging environment termination info and episode info into the current statistics.
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})

        # Update episode statistics: increment the number of episodes and episode length, and add the cumulative return of the current episode to the returns list
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        cur_returns.extend(episode_returns)

        # If in test mode and the specified number of test episodes is reached,
        # or if in training mode and the logging interval has been exceeded,
        # call the _log function to record the data.
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval: # Log the data to the logger
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch # Return the current batch data

    def _log(self, returns, stats, prefix): # Same as episode_runner
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x() # env_fn() is an environment function that creates and returns an environment instance
    while True:
        cmd, data = remote.recv() # Use remote.recv() to receive the command cmd and accompanying data from the main process
        if cmd == "step": # Receive the agents' actions, execute the environment's step function, and return state, reward, done flag, and other information
            actions = data
            # Take a step in the environment
            # Upon receiving the "step" command, parse data as agents' actions and execute these actions in the environment by calling env.step(actions)
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })

        # Send the current environment state, available actions for agents, observations, rewards, and termination status back to the main process via remote.send()
        elif cmd == "reset": # Reset the environment and return the initial state
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close": # Close the environment and terminate the process
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info": # Return the environment's metadata
            remote.send(env.get_env_info())
        elif cmd == "get_stats": # Return environment statistics data
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper(): # Wrapper class to handle serialization issues in parallel processing
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self): # The __getstate__ method serializes self.x using cloudpickle.dumps to handle complex objects like functions and classes
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob): # The __setstate__ method is used for deserializing ob, restoring self.x using pickle.loads
        import pickle
        self.x = pickle.loads(ob)

