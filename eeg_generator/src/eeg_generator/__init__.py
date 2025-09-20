from .generators import generate_eeg, EEGStream, log_calls
from .async_stream import eeg_producer, eeg_consumer, run_async_pubsub
from .real_data import load_real_eeg, filter_eeg
from .visualization import plot_eeg_samples, live_plot_stream

__version__ = "0.1.0"
__all__ = [
    "generate_eeg",
    "EEGStream",
    "log_calls",
    "eeg_producer",
    "eeg_consumer",
    "run_async_pubsub",
    "load_real_eeg",
    "filter_eeg",
    "plot_eeg_samples",
    "live_plot_stream"
]
