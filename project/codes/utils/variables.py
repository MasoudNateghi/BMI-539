# This file contains common variables that are used throughout the project.
# libraries
import wfdb
from utils.lpf import lp_filter_zero_phase

from project.codes.demo_load_data import fs_new

# variables
fs = 1000  # Sampling frequency
fc = 0.5  # Cut-off frequency
fs_new = 360  # New sampling frequency

# paths
run_local = True
if run_local:
    dataset_path = "/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/data/physionet.org/files/ptbdb/1.0.0"
    bw_path = "/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/data/physionet.org/files/nstdb/1.0.0/bw"
else:
    pass

bw = wfdb.rdrecord(bw_path).p_signal.T[0]
bw = lp_filter_zero_phase(bw, fc / fs_new)