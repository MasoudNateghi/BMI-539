import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.variables import *

from agent import CustomLSTMPolicy
from environment import ECGEnvironment


def evaluate_policy(model, env, n_eval_episodes=10):
    """Evaluate the trained policy"""
    rewards = []
    mse_values = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_mse = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            # Store MSE (negative of reward since reward is -MSE)
            episode_mse.append(-reward)

        rewards.append(episode_reward)
        mse_values.append(np.mean(episode_mse))

    return np.mean(rewards), np.std(rewards), np.mean(mse_values)


def train_model(channel_data, train_indices, baseline_wander,
                window_seconds=5, fs=fs, snr_db=10, total_timesteps=100000,
                eval_freq=10000):
    """Train model with monitoring"""
    # Create and wrap environment
    env = ECGEnvironment(
        channel_data=channel_data,
        train_indices=train_indices,
        baseline_wander=baseline_wander,
        window_seconds=window_seconds,
        fs=fs,
        snr_db=snr_db
    )
    env = DummyVecEnv([lambda: env])

    # Create evaluation environment
    eval_env = ECGEnvironment(
        channel_data=channel_data,
        train_indices=train_indices,
        baseline_wander=baseline_wander,
        window_seconds=window_seconds,
        fs=fs,
        snr_db=snr_db
    )
    eval_env = DummyVecEnv([lambda: eval_env])

    # Training history
    history = {
        'eval_rewards': [],
        'eval_mse': [],
        'value_loss': [],
        'policy_loss': [],
        'explained_variance': [],
        'kl_divergence': [],
        'timesteps': []
    }

    def callback(_locals, _globals):
        """Callback function to track metrics during training"""
        if len(history['timesteps']) == 0 or \
                _locals['self'].num_timesteps - history['timesteps'][-1] >= eval_freq:
            # Evaluate policy
            mean_reward, std_reward, mean_mse = evaluate_policy(
                _locals['self'], eval_env)

            # Store metrics
            history['timesteps'].append(_locals['self'].num_timesteps)
            history['eval_rewards'].append(mean_reward)
            history['eval_mse'].append(mean_mse)
            history['value_loss'].append(_locals['self'].logger.name_to_value['train/value_loss'])
            history['policy_loss'].append(_locals['self'].logger.name_to_value['train/policy_gradient_loss'])
            history['explained_variance'].append(_locals['self'].logger.name_to_value['train/explained_variance'])
            history['kl_divergence'].append(_locals['self'].logger.name_to_value['train/approx_kl'])

            # Print current metrics
            print(f"\nTimestep: {_locals['self'].num_timesteps}")
            print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"Mean MSE: {mean_mse:.6f}")
            print(f"Explained variance: {history['explained_variance'][-1]:.3f}")
            print(f"KL divergence: {history['kl_divergence'][-1]:.3f}\n")

        return True

    # Create and train model
    model = PPO(
        CustomLSTMPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model, history

