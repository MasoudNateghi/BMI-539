import pickle
import numpy as np

from utils.variables import *
from utils.dhandle import load_data
from pipeline.train import train_model

data = load_data(dataset_path)

nRecords = len(data[0])
train_size = int(0.8 * nRecords)
channel = 0  # Channel to train on

model, history = train_model(
    channel_data=data[channel],
    train_indices=list(np.arange(train_size)),
    baseline_wander=bw,
    window_seconds=5,
    fs=fs,
    snr_db=10,
    total_timesteps=30000,
    eval_freq=100
)

# save the model and history
model.save("misc/models/model")
with open('misc/models/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
