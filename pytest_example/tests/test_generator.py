from eeg.generator import generate_sample


def test_generate_sample():
    sample = generate_sample()
    assert len(sample) == 4
    for value in sample:
        assert isinstance(value, float)