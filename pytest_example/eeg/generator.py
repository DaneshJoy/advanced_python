import random


def generate_sample(n_channels: int = 4):
    # Generate a single sample
    return [random.gauss(0, 1) for _ in range(n_channels)]