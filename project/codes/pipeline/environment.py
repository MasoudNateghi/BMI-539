import gymnasium as gym
import numpy as np
from utils.lpf import lp_filter_zero_phase

class ECGEnvironment(gym.Env):
    def __init__(self, channel_data, fs, train_indices, baseline_wander, window_seconds=5, snr_db=10):
        """
        Args:
            channel_data: List of recordings for a specific channel
            train_indices: List of indices to use for training
            baseline_wander: Long baseline wander signal
            window_seconds: Window size in seconds
            fs: Sampling frequency in Hz
            snr_db: Target SNR in dB
        """
        super(ECGEnvironment, self).__init__()

        self.current_bw = None
        self.total_windows = None
        self.current_window = None
        self.noisy_signal = None
        self.clean_signal = None
        self.current_idx = None
        self.channel_data = channel_data
        self.train_indices = train_indices
        self.baseline_wander = baseline_wander
        self.fs = fs
        self.window_samples = int(window_seconds * fs)
        self.snr_db = snr_db

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0.01]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_samples,),
            dtype=np.float32
        )

    def get_random_baseline_segment(self, length):
        """Get random segment from long baseline wander signal"""
        if length > len(self.baseline_wander):
            raise ValueError("Recording length is longer than baseline wander signal")

        max_start = len(self.baseline_wander) - length
        start_idx = np.random.randint(0, max_start)
        return self.baseline_wander[start_idx:start_idx + length]

    def add_baseline_wander(self, clean_signal):
        """Add random baseline wander segment at specified SNR"""
        # Get random segment of baseline wander
        noise = self.get_random_baseline_segment(len(clean_signal))

        # Calculate scaling factor for desired SNR
        clean_power = np.mean(clean_signal ** 2)
        noise_power = np.mean(noise ** 2)
        target_noise_power = clean_power / (10 ** (self.snr_db / 10))
        scaling_factor = np.sqrt(target_noise_power / noise_power)

        # Add scaled baseline wander
        noisy_signal = clean_signal + scaling_factor * noise
        return noisy_signal, scaling_factor * noise

    def reset(self, seed=None, options=None):
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Randomly select a recording
        self.current_idx = np.random.choice(self.train_indices)
        self.clean_signal = self.channel_data[self.current_idx]

        # Add random segment of baseline wander
        self.noisy_signal, self.current_bw = self.add_baseline_wander(self.clean_signal)

        # Initialize window counter
        self.current_window = 0
        self.total_windows = len(self.clean_signal) // self.window_samples

        # Reset LSTM states when starting new episode
        if hasattr(self, 'policy'):
            self.policy.reset_hidden_states()

        # Return first window
        return self.noisy_signal[:self.window_samples].astype(np.float32), {}

    def step(self, action):
        # Get current window indices
        start_idx = self.current_window * self.window_samples
        end_idx = start_idx + self.window_samples

        # Get filter parameters
        cutt_off = action[0]

        # Apply filter
        filtered_signal = self.noisy_signal[start_idx:end_idx] - lp_filter_zero_phase(self.noisy_signal[start_idx:end_idx], cutt_off / self.fs)

        # Calculate reward (negative MSE)
        clean_segment = self.clean_signal[start_idx:end_idx]
        reward = -np.mean((filtered_signal - clean_segment) ** 2)

        # Move to next window
        self.current_window += 1
        truncated = self.current_window >= self.total_windows

        # Get next state
        if not truncated:
            next_start = self.current_window * self.window_samples
            next_state = self.noisy_signal[next_start:next_start + self.window_samples]
        else:
            next_state = np.zeros(self.window_samples)

        info = {
            'filtered_signal': filtered_signal,
            'clean_signal': clean_segment,
            'noisy_signal': self.noisy_signal[start_idx:end_idx],
            'fc': cutt_off
        }

        return next_state.astype(np.float32), reward, truncated, {}, info