import numpy as np
from pipeline.environment import ECGEnvironment

class NoisyECGEnvironment(ECGEnvironment):
    def __init__(self, noisy_signal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_windows = None
        self.current_window = None
        self.clean_signal = None
        self.current_idx = None
        self.noisy_signal = noisy_signal  # Pre-noisy signal provided externally

    def reset(self, seed=None, options=None):
        """Custom reset method for pre-noisy signals."""
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Use the pre-noisy signal
        self.current_idx = 0  # Since it's a single signal, index is 0
        self.clean_signal = self.channel_data[self.current_idx]  # Clean ECG signal for reward calculation

        # No additional noise added
        self.current_window = 0
        self.total_windows = len(self.clean_signal) // self.window_samples

        # Reset LSTM states when starting a new episode
        if hasattr(self, 'policy'):
            self.policy.reset_hidden_states()

        # Return the first window of the noisy signal
        return self.noisy_signal[:self.window_samples].astype(np.float32), {}


def run_inference_on_ecg(model, env):
    obs = env.reset()[0]
    done = False
    episode_rewards = []
    fc_values = []
    filtered_signals = []
    clean_signals = []
    noisy_signals = []

    while not done:
        # Predict action (cutoff frequency)
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, done, _, info = env.step(action)

        # Log metrics and signals
        episode_rewards.append(reward)
        fc_values.append(info['fc'])  # Extract cutoff frequency
        filtered_signals.append(info['filtered_signal'])
        clean_signals.append(info['clean_signal'])
        noisy_signals.append(info['noisy_signal'])

    return episode_rewards, fc_values, filtered_signals, clean_signals, noisy_signals


def get_random_baseline_segment(signal_length, baseline_wander):
    """Get random segment from long baseline wander signal"""
    if signal_length > len(baseline_wander):
        raise ValueError("Recording length is longer than baseline wander signal")

    max_start = len(baseline_wander) - signal_length
    start_idx = np.random.randint(0, max_start)
    return baseline_wander[start_idx:start_idx + signal_length]

def add_baseline_wander(clean_signal, baseline_wander, snr_db=10):
    """Add random baseline wander segment at specified SNR"""
    # Get random segment of baseline wander
    noise = get_random_baseline_segment(len(clean_signal), baseline_wander)

    # Calculate scaling factor for desired SNR
    clean_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)

    # Add scaled baseline wander
    noisy_signal = clean_signal + scaling_factor * noise
    return noisy_signal
