# This file contains common variables that are used throughout the project.
# libraries
import os
import wfdb
from utils.lpf import lp_filter_zero_phase

# paths and directories
run_local = False
if run_local:
    dataset_path = "/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/data/physionet.org/files/ptbdb/1.0.0"
    bw_path = "/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/data/physionet.org/files/nstdb/1.0.0/bw"
else:
    dataset_path = "/labs/samenilab/team/masoud_nateghi/data/physionet.org/files/ptbdb/1.0.0/"
    bw_path = "/labs/samenilab/team/masoud_nateghi/data/physionet.org/files/nstdb/1.0.0/bw"

os.makedirs("misc/models", exist_ok=True)
os.makedirs("misc/results", exist_ok=True)
os.makedirs("misc/dataset", exist_ok=True)

# variables
fs_old = 1000  # Sampling frequency
fc = 0.5  # Cut-off frequency
fs = 360  # New sampling frequency
bw = wfdb.rdrecord(bw_path).p_signal.T[0]  # read bw
bw = lp_filter_zero_phase(bw, fc / fs)  # remove high frquency noise from bw