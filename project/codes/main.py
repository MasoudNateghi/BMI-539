import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.variables import *
from utils.processing import preproc
from utils.dhandle import find_paths, extract_dataset

from train import train_model

# Prepare data
paths = find_paths(dataset_path)
records = extract_dataset(paths)
nRecords = len(records)
nChannels = records[0].shape[0]

# Creating ground truth by an aggressive low-pass filtering
records = Parallel(n_jobs=-1)(
    delayed(preproc)(record, fc, fs, fs_old, order=64, Q_factor=30, freq=50)
    for record in tqdm(records, desc="Preprocessing ECG signals: ")
)

# Separating channels across records in a dictionary
data = {}  # Dictionary to store channels
for record in records:
    for channel in range(nChannels):
        data[channel] = data.get(channel, []) + [record[channel]]


train_size = int(0.8 * nRecords)

model, history = train_model(
    channel_data=data[0],
    train_indices=list(np.arange(train_size)),
    baseline_wander=bw,
    window_seconds=5,
    fs=fs,
    snr_db=10,
    total_timesteps=10000,  # Increase training time
    eval_freq=1000  # Evaluate every 10000 steps
)
