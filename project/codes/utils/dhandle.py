# This file contains functions to load data.
import os
import wfdb
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.variables import *
from utils.processing import preproc

from project.codes.utils.variables import dataset_path


def find_paths(dataset_path):
    paths = []
    for folder in sorted(os.listdir(dataset_path)):
        data_folder = os.path.join(dataset_path, folder)
        if os.path.isdir(data_folder):
            for file in sorted(os.listdir(data_folder)):
                if file.endswith(".dat"):
                    file, ext = os.path.splitext(file)
                    data_file = os.path.join(dataset_path, folder, file)
                    paths.append(data_file)
    return sorted(paths)


def read_ecg_data(root, channels=None):
    record = wfdb.rdrecord(root, channels=channels)
    ecg_data = record.p_signal.T
    return ecg_data


def extract_dataset(paths, channels=None):
    records = []
    for path in tqdm(paths, desc="Reading Records: "):
        ecg_data = read_ecg_data(path, channels)
        records.append(ecg_data)

    return records

def load_data(data_path=dataset_path):
    if os.path.exists("misc/dataset/data.pkl"):
        print("Dataset already exists. Loading...")
        with open("misc/dataset/data.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        print("Dataset not found. Creating...")

        # Prepare data
        paths = find_paths(data_path)
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

    return data
