from .generators import generate_eeg, EEGStream, log_calls
from .async_stream import eeg_producer, eeg_consumer, run_async_pubsub
from .real_data import load_real_eeg, filter_eeg
from .visualization import plot_eeg_samples, live_plot_stream
from .multithread import EEGProducer, live_consumer, queue_stream

# import logging
# from .logging import get_logger, setup_logging

# # Package-wide logger; lazy default (quiet unless configured)
# logger = get_logger()
# logger.addHandler(logging.NullHandler())

# # Optional: auto-setup default logging (console only, INFO level)
# # Uncomment if you want the package to always log by default
# # setup_logging(level=logging.INFO, log_to_console=True)


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
    "live_plot_stream",
    "EEGProducer",
    "live_consumer",
    "queue_stream",
]
