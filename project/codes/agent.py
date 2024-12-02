import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomLSTMPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        self.prev_action = None
        self.hidden_state = None
        self.segment_size = observation_space.shape[0]

        # LSTM network
        self.lstm = nn.LSTM(
            input_size=self.segment_size + 1,  # signal + previous action
            hidden_size=64,
            batch_first=True
        )

        # Action mean and log std
        self.action_mean = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Log standard deviation parameter
        self.log_std = nn.Parameter(torch.zeros(1))

        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize states
        self.reset_hidden_states()

    def forward(self, obs, deterministic=False):
        if self.prev_action is None:
            self.prev_action = torch.zeros((obs.shape[0], 1), device=obs.device)

        # Combine observation and previous action
        x = torch.cat([obs, self.prev_action], dim=1)
        x = x.unsqueeze(1)  # Add sequence dimension

        # LSTM forward pass
        lstm_out, self.hidden_state = self.lstm(x, self.hidden_state)
        features = lstm_out[:, -1, :]  # Take last timestep

        # Get action mean and std
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.log_std)

        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()

        # Clip action between 0 and 1
        action = torch.clamp(action, 0, 1)

        # Get value
        value = self.value_net(features)

        # Get log probability
        log_prob = dist.log_prob(action)

        # Update previous action
        self.prev_action = action.detach()

        return action, value, log_prob

    def reset_hidden_states(self):
        """Reset LSTM hidden states and previous action between episodes"""
        self.prev_action = None
        self.hidden_state = None
