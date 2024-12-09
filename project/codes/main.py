import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from pipeline.train import train_model
from utils.variables import *
from utils.processing import preproc
from utils.dhandle import find_paths, extract_dataset

if os.path.exists("misc/dataset/data.pkl"):
    print("Dataset already exists. Loading...")
    with open("misc/dataset/data.pkl", "rb") as f:
        data = pickle.load(f)
else:
    print("Dataset not found. Creating...")

    # Prepare data
    paths = find_paths(dataset_path)
    records = extract_dataset(paths)
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

    # Save the data
    with open("misc/dataset/data.pkl", "wb") as f:
        pickle.dump(data, f)

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
