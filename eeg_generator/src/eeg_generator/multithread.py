import time
import queue
import random
import threading
import numpy as np

from eeg_generator.visualization import plot_eeg_samples, live_plot_stream


class EEGProducer(threading.Thread):
    def __init__(self, n_channel=4, sampling_rate=10, duration=5, out_queues=None, simulate_class=None):
        super().__init__()
        self.n_channel = n_channel
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.out_queues = out_queues or [queue.Queue()]
        self._running = threading.Event()
        self._running.set()
        self.simulate_class = simulate_class  # None: random, 0: 'left', 1: 'right'

    def run(self):
        n_samples = self.sampling_rate * self.duration
        t = np.linspace(0, self.duration, n_samples)
        for i in range(n_samples):
            if not self._running.is_set():
                break
            if self.simulate_class == 0:  # Simulate 'left' with 10Hz in channel 0-1
                sample = [np.sin(2 * np.pi * 10 * t[i]) + random.gauss(0, 0.1) for _ in range(self.n_channel)]
                sample[2:] = [random.gauss(0, 1) for _ in range(2)]  # Noise in others
            elif self.simulate_class == 1:  # Simulate 'right' with 12Hz in channel 2-3
                sample = [np.sin(2 * np.pi * 12 * t[i]) + random.gauss(0, 0.1) for _ in range(self.n_channel)]
                sample[:2] = [random.gauss(0, 1) for _ in range(2)]  # Noise in others
            else:  # Random
                sample = [random.gauss(0, 1) for _ in range(self.n_channel)]
            for q in self.out_queues:
                q.put(sample)
            time.sleep(1 / self.sampling_rate)
        for q in self.out_queues:
            q.put(None)


def live_consumer(in_queue: queue.Queue, n_samples: int = 10):
    collected_samples = []
    for _ in range(n_samples):
        try:
            sample = in_queue.get(timeout=1)
            collected_samples.append(sample)
            if sample is None:
                break
        except queue.Empty:
            break
    return collected_samples


def queue_stream(q: queue.Queue):
    while True:
        try:
            sample = q.get(timeout=1)  # Add timeout to prevent indefinite blocking
            if sample is None:
                break
            yield sample
        except queue.Empty:
            break

if __name__ == "__main__":
    q_consume = queue.Queue()
    q_plot = queue.Queue()
    producer = EEGProducer(n_channels=4, sampling_rate=10, duration=3, out_queues=[q_consume, q_plot])

    producer.start()
    
    samples = eeg_consumer(q_consume, n_samples=5)
    fig = plot_eeg_samples(samples, sampling_rate=10)
    print(f'Collected {len(samples)} EEG samples')
    
    stream = queue_stream(q_plot)
    live_plot_stream(stream, n_channels=4, sampling_rate=10)

    producer.join()
    

