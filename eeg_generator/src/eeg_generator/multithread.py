import time
import queue
import random
import threading

from eeg_generator.visualization import plot_eeg_samples, live_plot_stream


class EEGProducer(threading.Thread):
    def __init__(self, n_channels: int = 4, sampling_rate: int = 10, duration: int = 1,
                 out_queues: list[queue.Queue] = None):
        super().__init__()
        self.out_queues = out_queues or [queue.Queue()]
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.duration = duration
        self._running = threading.Event()
        self._running.set()

    def run(self):
        n_samples = int(self.duration * self.sampling_rate)
        for _ in range(n_samples):
            if not self._running.is_set():
                break
            sample = [random.gauss(0, 1) for _ in range(self.n_channels)]
            for out_queue in self.out_queues:
                out_queue.put(sample)
            time.sleep(1 / self.sampling_rate)
        # Send termination signal to all queues
        for out_queue in self.out_queues:
            out_queue.put(None)

    def stop(self):
        self._running.clear()


def eeg_consumer(in_queue: queue.Queue, n_samples: int = 10):
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
    

