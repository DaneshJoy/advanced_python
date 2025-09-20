import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Iterable


def plot_eeg_samples(
    samples: List[List[float]],
    sampling_rate: int,
    channel_labels: Optional[List[str]] = None
):
    n_channels = len(samples[0])
    n_samples_total = len(samples)
    time = np.arange(n_samples_total) / sampling_rate
    
    if channel_labels is None:
        channel_labels = [f'Ch-{i + 1}' for i in range(n_channels)]
    
    data_array = np.array(samples)
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2*n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    for ch in range(n_channels):
        axes[ch].plot(time, data_array[:, ch])
        axes[ch].set_ylabel(channel_labels[ch])
        axes[ch].grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return fig


def live_plot_stream(
    stream: Iterable[List[float]],
    sampling_rate: int,
    n_channels: int,
    buffer_size: int = 100
):

    channel_labels = [f'Ch-{i + 1}' for i in range(n_channels)]

    fig, axes = plt.subplots(len(channel_labels), 1, figsize=(10, 2*len(channel_labels)), sharex=True)
    
    # Create the initial buffer
    data = np.zeros((n_channels, buffer_size))
    times = np.arange(buffer_size) / sampling_rate

    lines = [axes[ch].plot(times, data[ch])[0] for ch in range(n_channels)]

    for ch in range(n_channels):
        axes[ch].set_ylabel(channel_labels[ch])
        axes[ch].grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    sample_count = 0
    for sample in stream:
        data = np.roll(data, -1, axis=1)
        current_time_end = sample_count / sampling_rate
        times = np.linspace(
            max(0, current_time_end - buffer_size / sampling_rate),
            current_time_end,
            buffer_size)
        for ch in range(n_channels):
            data[ch, -1] = sample[ch]
            lines[ch].set_data(times, data[ch])
            axes[ch].relim()
            axes[ch].autoscale_view()
            
        sample_count += 1
        
        plt.draw()
        plt.pause(0.01)
    # plt.show()
