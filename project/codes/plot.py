import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load history
with open('misc/models/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# plot and save figures
plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['eval_rewards'])
plt.xlabel('Episodes', fontsize=25)
plt.ylabel('Reward', fontsize=25)
plt.savefig("misc/results/average_reward.png")
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['eval_mse'])
plt.xlabel('Episodes', fontsize=25)
plt.ylabel('MSE', fontsize=25)
plt.savefig("misc/results/mean_square_error.png")
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['eval_snr'])
plt.xlabel('Episodes', fontsize=25)
plt.ylabel('SNR (dB)', fontsize=25)
plt.savefig("misc/results/snr.png")
plt.close()

data = np.concatenate(history['fc_values'])
window_size = 1000
smoothened_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')

plt.figure(figsize=(8, 6))
plt.plot(data)
plt.plot(smoothened_data)
plt.xlabel('Segments', fontsize=25)
plt.ylabel('Frequency', fontsize=25)
plt.savefig("misc/results/fc_values.png")
plt.close()
