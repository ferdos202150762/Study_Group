from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        # Calculate the input shape for each agent, e.g., the size of the observation space, and assign it to input_shape
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        # Get the agent output type from args (e.g., action probability distribution or Q-values)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        # Initialize hidden_states as None, which will be used to store hidden states (for RNNs, etc.)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep] # Get available actions at current time step t_ep from ep_batch
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode) # Get agent outputs at the current time step

        # Use the action selector to choose actions from agent outputs, based on available actions and environment time step t_env
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)  # Call _build_inputs() to construct agent inputs at time step t
        avail_actions = ep_batch["avail_actions"][:, t]  # Get available actions at time step t
        # Use the agent network to compute outputs given the inputs and current hidden states, and update hidden states
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Check if the agent output type is policy logits (action probabilities); if so, apply softmax
        if self.agent_output_type == "pi_logits":

            # If mask_before_softmax is enabled, set logits of unavailable actions to a very negative value to ensure they get zero probability after softmax
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            # Apply softmax to convert to action probabilities
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            # If not in test mode, use epsilon-greedy strategy for action selection
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                # During exploration, the output is a weighted sum of original output and uniform random actions
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        # Initialize hidden states so that each agent in each batch has its own hidden state
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self): # Return the parameters of the agent
        return self.agent.parameters()

    def load_state(self, other_mac): # Load state from another MAC
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self): # Move the model to GPU
        self.agent.cuda()

    def save_models(self, path): # Save model parameters
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path): # Load model parameters
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape): # Build the agent based on the registry and input shape
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t): # Construct inputs, including observation, previous actions (if applicable), and agent IDs
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme): # Compute the shape of the input, including observation, action encoding, and agent ID
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
