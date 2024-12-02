import pickle
import matplotlib.pyplot as plt

# Load history
with open('models/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# plot and save figures
plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['eval_rewards'])
plt.xlabel('Timesteps', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.savefig("results/average_reward.png")
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['eval_mse'])
plt.xlabel('Timesteps', fontsize=15)
plt.ylabel('MSE', fontsize=15)
plt.savefig("results/mean_square_error.png")
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['eval_snr'])
plt.xlabel('Timesteps', fontsize=15)
plt.ylabel('SNR (dB)', fontsize=15)
plt.savefig("results/snr.png")
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['value_loss'], label='Value Loss')
plt.plot(history['timesteps'], history['policy_loss'], label='Policy Loss')
plt.xlabel('Timesteps', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend()
plt.savefig("results/training_losses.png")
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(history['timesteps'], history['kl_divergence'], label='KL Divergence')
plt.plot(history['timesteps'], history['explained_variance'], label='Explained Variance')
plt.xlabel('Timesteps', fontsize=15)
plt.ylabel('Value', fontsize=15)
plt.legend()
plt.savefig("results/kl_divergence_explained_variance.png")
plt.close()
