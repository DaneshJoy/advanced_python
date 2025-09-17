import time
import random
from functools import wraps


def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        return result
    return wrapper


@log_calls
def generate_eeg(n_channel: int = 4, sampling_rate: int = 10, duration: int = 1):
    n_samples = sampling_rate * duration
    for _ in range(n_samples):
        channel_data = [random.gauss(0, 1) for _ in range(n_channel)]
        yield channel_data
        yield '-'*35
        time.sleep(1/sampling_rate)


class EEGStream:
    def __init__(self, n_channel: int = 4, sampling_rate: int = 10, duration: int = 1):
        self.n_channel = n_channel
        self.sampling_rate = sampling_rate
        self.duration = duration
        self._running = False
        self._total_samples = sampling_rate * duration
        self._sample_count = 0

    def __iter__(self):
        self._running = True
        self._sample_count = 0
        return self

    def __next__(self):
        if not self._running or self._sample_count >= self._total_samples:
            raise StopIteration('The generator is stopped')
        n_samples = self.sampling_rate * self.duration
        sample = [random.gauss(0, 1) for _ in range(self.n_channel)]
        time.sleep(1/self.sampling_rate)
        self._sample_count += 1
        return sample

    def stop(self):
        self._running = False