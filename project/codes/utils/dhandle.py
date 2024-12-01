# This file contains functions to load data.
import os
import wfdb
from tqdm import tqdm


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
