

# **Python Testing with `pytest`: A Mini Tutorial**

Here’s a compact tutorial using your code as the basis to teach **Python testing with `pytest`**, including unit tests, threading, async, plotting, and mocking:

This tutorial shows how to test different types of Python code using `pytest`, from simple functions to async and threaded code.

------

## **1. Testing Simple Functions**

```python
from eeg.generator import generate_sample

def test_generate_sample():
    sample = generate_sample()
    assert len(sample) == 4                  # check output length
    for value in sample:
        assert isinstance(value, float)      # check value types
```

- **Key concepts**: `assert` statements, checking types and values.
- **Run**: `pytest test_file.py`

------

## **2. Testing Threaded Code**

```python
import queue
import threading
from eeg.threaded import threaded_producer, threaded_consumer

def test_threaded_producer_consumer():
    q = queue.Queue()
    n = 4

    producer_thread = threading.Thread(target=threaded_producer, args=(q, n))
    producer_thread.start()
    producer_thread.join()
    
    result = threaded_consumer(q, n)
    assert q.get() is None
    assert result == list(range(n))
```

- Threads can be started and joined.
- Use a `Queue` to safely communicate between threads.
- Always check final queue state and results.

**Other threaded variations:**

- Test empty queue.
- Use a `ThreadPoolExecutor`.
- Wrap consumers in threads.

------

## **3. Testing Async Code**

```python
import pytest
from eeg.async_producer import async_producer

@pytest.mark.asyncio
async def test_async_producer():
    result = await async_producer(4)
    assert result == [0, 1, 2, 3]
    assert len(result) == 4
```

- **Key concepts**: `@pytest.mark.asyncio`, `await`.
- Async tests must be marked with `asyncio` to run properly.

------

## **4. Testing Plotting Functions**

```python
import matplotlib.pyplot as plt
import numpy as np
from eeg_generator import plot_eeg_samples

@pytest.fixture
def dummy_samples():
    return [[np.random.normal(0, 1) for _ in range(4)] for _ in range(20)]

def test_plot_eeg_samples(dummy_samples, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # prevent GUI
    fig = plot_eeg_samples(dummy_samples)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
```

- **Key concepts**: `monkeypatch` to mock functions, `pytest.fixture` for reusable data.

------

## **5. Mocking Time and External Calls**

```python
from unittest.mock import patch
import time
from eeg.generator import generate_eeg

@patch('time.sleep')
def test_generate_eeg_timing(mock_sleep):
    sampling_rate = 10
    duration = 1
    list(generate_eeg(sampling_rate=sampling_rate, duration=duration))
    
    assert mock_sleep.call_count == sampling_rate * duration
    mock_sleep.assert_called_with(1/sampling_rate)
```

- **Use case**: Avoid delays or external dependencies.
- `unittest.mock.patch` replaces a function temporarily.

**Alternatives:**

- `monkeypatch.setattr(time, "sleep", lambda _: None)`:

  - ```python
    def test_generate_eeg_with_monkeypatch(monkeypatch):
        # patch time.sleep just inside this test
        monkeypatch.setattr(time, "sleep", lambda _: None)
    
        sampling_rate = 10
        duration = 1
        list(generate_eeg(sampling_rate=sampling_rate, duration=duration))
    ```

- `pytest-mock`'s `mocker.patch("time.sleep")`:

  -  requires `pytest-mock`

  - ```python
    def test_generate_eeg_with_mocker(mocker):
        spy_sleep = mocker.patch("time.sleep")  # behaves like unittest.patch
        
        sampling_rate = 10
        duration = 1
        list(generate_eeg(sampling_rate=sampling_rate, duration=duration))
        
        assert spy_sleep.call_count == sampling_rate * duration
        spy_sleep.assert_called_with(1 / sampling_rate)
    ```

  - 

------

## **6. Handling Exceptions**

```python
import pytest
from eeg.generator import generate_eeg

def test_generate_eeg_invalid_inputs():
    with pytest.raises(TypeError):
        list(generate_eeg(n_channel="invalid"))
```

- `pytest.raises` checks that a specific exception is thrown.

------

## ✅ **Summary of Concepts Covered**

- **Basic assertions** for outputs and types.
- **Threading tests** using `Queue` and `Thread`.
- **Async tests** using `@pytest.mark.asyncio`.
- **Plotting tests** using `monkeypatch` to avoid GUI.
- **Mocking** with `patch`, `monkeypatch`, and `pytest-mock`.
- **Exception testing** with `pytest.raises`.
- **Fixtures** for reusable test data.

------

This example set makes `pytest` practical for **scientific code**, **EEG simulation**, or any **producer/consumer pipelines**.

