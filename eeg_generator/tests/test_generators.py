import pytest
from unittest.mock import patch
from eeg_generator import generate_eeg, EEGStream


def test_generate_eeg_output_length():
    n_channel = 4
    sampling_rate = 10
    duration = 1
    samples = list(generate_eeg(n_channel, sampling_rate, duration))
    assert len(samples) == sampling_rate * duration, "Incorrect number of samples generated"
    assert all(len(sample) == n_channel for sample in samples), "Incorrect number of channels in samples"

def test_generate_eeg_data_distribution():
    n_channel = 4
    sampling_rate = 10
    duration = 1
    samples = list(generate_eeg(n_channel, sampling_rate, duration))
    all_data = [value for sample in samples for value in sample]
    assert len(all_data) == n_channel * sampling_rate * duration, "Incorrect total data points"
    assert all(isinstance(value, float) for value in all_data), "Data points should be floats"

@patch('time.sleep')  # Mock time.sleep to avoid actual delays
def test_generate_eeg_timing(mock_sleep):
    sampling_rate = 10
    duration = 1
    list(generate_eeg(sampling_rate=sampling_rate, duration=duration))
    assert mock_sleep.call_count == sampling_rate * duration, "Incorrect number of sleep calls"
    mock_sleep.assert_called_with(1/sampling_rate)

def test_eeg_stream_init():
    stream = EEGStream(n_channel=4, sampling_rate=10, duration=1)
    assert stream.n_channel == 4, "Incorrect number of channels"
    assert stream.sampling_rate == 10, "Incorrect sampling rate"
    assert stream.duration == 1, "Incorrect duration"
    assert stream._running is False, "Stream should not be running initially"
    assert stream._total_samples == 10, "Incorrect total samples"
    assert stream._sample_count == 0, "Sample count should be zero initially"

def test_eeg_stream_iteration():
    stream = EEGStream(n_channel=4, sampling_rate=10, duration=1)
    samples = list(stream)
    assert len(samples) == 10, "Incorrect number of samples in iteration"
    assert all(len(sample) == 4 for sample in samples), "Incorrect number of channels in samples"
    assert stream._sample_count == 10, "Sample count not updated correctly"

def test_eeg_stream_stop():
    stream = EEGStream(n_channel=4, sampling_rate=10, duration=1)
    stream.__iter__()  # Start iteration
    stream.stop()
    assert stream._running is False, "Stream should be stopped after calling stop"
    with pytest.raises(StopIteration, match="The generator is stopped"):
        next(stream)

@patch('time.sleep')
def test_eeg_stream_timing(mock_sleep):
    stream = EEGStream(n_channel=4, sampling_rate=10, duration=1)
    list(stream)
    assert mock_sleep.call_count == 10, "Incorrect number of sleep calls in stream"
    mock_sleep.assert_called_with(1/10)

def test_eeg_stream_early_stop():
    stream = EEGStream(n_channel=4, sampling_rate=10, duration=1)
    iterator = iter(stream)
    next(iterator)  # Get one sample
    stream.stop()
    with pytest.raises(StopIteration, match="The generator is stopped"):
        next(iterator)
    assert stream._sample_count == 1, "Sample count should reflect early stop"

def test_generate_eeg_invalid_inputs():
    with pytest.raises(TypeError):
        list(generate_eeg(n_channel="invalid"))

def test_eeg_stream_invalid_inputs():
    with pytest.raises(TypeError):
        stream = EEGStream(n_channel="invalid")
        for _ in stream:
            pass
        
