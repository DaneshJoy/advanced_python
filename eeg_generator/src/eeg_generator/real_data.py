"""Loading and processing real EEG datasets using MNE."""

import mne
from typing import Tuple, Optional


def load_real_eeg(
    subjects: int = 1, runs: list = None, preload: bool = True
) -> mne.io.Raw:
    """Load real EEG data from MNE's EEGBCI dataset."""
    if runs is None:
        runs = [6, 10]  # Runs for left vs right fist motor imagery
    patient_data = mne.datasets.eegbci.load_data(subjects=subjects, runs=runs)
    raws = [mne.io.read_raw_edf(f, preload=preload) for f in patient_data]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)  # Standardize channel names
    raw.set_montage('standard_1020')
    return raw


def filter_eeg(raw: mne.io.Raw, l_freq: float = 7, h_freq: float = 30) -> mne.io.Raw:
    """Filter EEG data (low-pass/high-pass)."""
    filtered = raw.copy().pick("eeg").filter(l_freq=l_freq, h_freq=h_freq)
    return filtered


def plot_real_eeg(raw: mne.io.Raw, block: bool = True):
    """Plot real EEG data using MNE's built-in plotter."""
    raw.plot(block=block)


def plot_real_eeg_simple(raw, channel=0, start=0, duration=5):
    import matplotlib.pyplot as plt

    """
    Plot EEG signals using matplotlib.
    start: start time in seconds
    duration: duration in seconds
    """
    data, times = raw.get_data(return_times=True)
    sfreq = raw.info["sfreq"]

    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)

    plt.figure(figsize=(12, 3))
    plt.plot(
        times[start_idx:end_idx],
        data[channel, start_idx:end_idx],
        label=raw.ch_names[channel],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.title("EEG Signals (Simple Plot)")
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # Example usage
    raw = load_real_eeg(subjects=1, runs=[1])
    filtered = filter_eeg(raw, 1)
    plot_real_eeg_simple(filtered, 1, start=0, duration=10)
