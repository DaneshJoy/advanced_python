import mne
import matplotlib.pyplot as plt


def load_real_eeg(subjects: int = 1, runs: list = None, preload: bool = True):
    patient_data = mne.datasets.eegbci.load_data(subjects=subjects, runs=runs)
    raw_data = mne.io.read_raw_edf(patient_data[0], preload=preload)
    return raw_data


def filter_eeg(raw, l_freq: float = 0.5, h_freq: float = 40):
    filtered = raw.copy().pick("eeg").filter(l_freq=l_freq, h_freq=h_freq)
    return filtered
