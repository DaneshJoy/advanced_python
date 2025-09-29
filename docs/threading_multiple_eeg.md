**Threads can run independent tasks in parallel**.
We’ll spin up **multiple EEGProducer threads**, each with different parameters (e.g. channels, sampling rate, noise variance), and let them run *at the same time*.

------

## 🧵 Example: Multiple EEG Producers in Parallel

```python
import threading
import queue
import time
import random
import numpy as np
import matplotlib.pyplot as plt

class EEGProducer(threading.Thread):
    def __init__(self, name, n_channel=4, sampling_rate=10, duration=3, noise=1.0, out_queue=None):
        super().__init__(name=name)
        self.n_channel = n_channel
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.noise = noise
        self.out_queue = out_queue or queue.Queue()
        self._running = threading.Event()
        self._running.set()

    def run(self):
        n_samples = self.sampling_rate * self.duration
        for _ in range(n_samples):
            if not self._running.is_set():
                break
            sample = [random.gauss(0, self.noise) for _ in range(self.n_channel)]
            timestamp = time.time()
            # store tuple (source, timestamp, sample)
            self.out_queue.put((self.name, timestamp, sample))
            time.sleep(1/self.sampling_rate)

    def stop(self):
        self._running.clear()
```

------

### Start Multiple Producers

```python
print("===== Multiple EEG Producers =====")

out_q = queue.Queue()

# Different properties
producers = [
    EEGProducer("FastHighNoise", n_channel=2, sampling_rate=20, duration=2, noise=2.0, out_queue=out_q),
    EEGProducer("SlowLowNoise", n_channel=4, sampling_rate=5, duration=4, noise=0.5, out_queue=out_q),
    EEGProducer("Default", n_channel=3, sampling_rate=10, duration=3, noise=1.0, out_queue=out_q),
]

# Start all
for p in producers:
    p.start()
```

------

### Consume from Shared Queue

```python
collected = []
active = len(producers)

while active > 0:
    try:
        src, ts, sample = out_q.get(timeout=1)
        print(f"[{src}] @ {ts:.2f}: {sample}")
        collected.append((src, ts, sample))
    except queue.Empty:
        # check if all producers are done
        active = sum(1 for p in producers if p.is_alive())
```

------

### Wait for Completion

```python
for p in producers:
    p.join()

print(f"Total collected: {len(collected)} samples")
```

------

### Grouped Plot

We can separate by **signal source**:

```python
from collections import defaultdict

def plot_grouped(data, sampling_rate_lookup):
    grouped = defaultdict(list)
    for src, ts, sample in data:
        grouped[src].append(sample)

    for src, samples in grouped.items():
        fig = plot_eeg_samples(samples, sampling_rate=sampling_rate_lookup[src])
        fig.suptitle(f"EEG from {src}")

sampling_rate_lookup = {p.name: p.sampling_rate for p in producers}
plot_grouped(collected, sampling_rate_lookup)
```

------

## 🧠 Tips

- **Each thread is independent** and can have its own speed & signal properties.
- **Coordination via a single queue** (merging multiple streams).
- The **consumer doesn’t know which producer is next** → arrival order depends on timing.

------

