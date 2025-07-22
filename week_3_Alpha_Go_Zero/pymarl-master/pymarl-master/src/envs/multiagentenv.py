class MultiAgentEnv(object):

    def step(self, actions):                               # Receives actions from agents and performs the next environment step.
        """ Returns reward, terminated, info """           # This is the core function in reinforcement learning; it typically updates the environment state based on the input actions.
        raise NotImplementedError

    def get_obs(self):                                     # Returns the observations of all agents.
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):                     # Returns the observation of the specified agent by agent_id.
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):                                # Returns the shape (size) of a single agent's observation.
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):                                   # Returns the global state of the environment.
        raise NotImplementedError

    def get_state_size(self):                              # Returns the shape (size) of the global state.
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):                           # Returns the available actions for all agents.
        # This is important in multi-agent scenarios, as different agents may have different available actions in certain situations.
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):           # Returns the available actions for the specified agent.
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):                           # Returns the total number of actions an agent can take.
        # This is typically used in discrete action spaces. For continuous spaces, this method might need to be adapted.
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):                                        # Reinitializes the environment and returns initial observations and state.
        # This function is called at the beginning of each new training episode.
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):                                       # Visualizes the current state of the environment.
        raise NotImplementedError

    def close(self):                                        # Closes the environment and releases resources after training ends.
        raise NotImplementedError

    def seed(self):                                         # Sets the random seed for reproducibility.
        raise NotImplementedError

    def save_replay(self):                                  # Saves a replay of the episode, recording agent trajectories.
        raise NotImplementedError

    def get_env_info(self):                                 # Returns environment-related information as a dictionary.
        env_info = {"state_shape": self.get_state_size(),   # Dimension of the global state
                    "obs_shape": self.get_obs_size(),       # Dimension of each agent's observation
                    "n_actions": self.get_total_actions(),  # Number of possible actions per agent
                    "n_agents": self.n_agents,              # Number of agents
                    "episode_limit": self.episode_limit}    # Maximum number of steps per episode
        return env_info
