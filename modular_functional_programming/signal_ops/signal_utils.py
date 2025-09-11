
import numpy as np


def make_signal(signal_type, signal_range):
    """
    Create a signal.

    Args:
        signal_type (str): The type of the signal ('sin' or 'cos').
        signal_range (tuple): The range of x values for the signal.

    Returns:
        tuple: The x and y values of the signal.
    """
    x = np.linspace(signal_range[0], signal_range[1], 1000)
    y = None
    if signal_type == 'sin':
        y = np.sin(x)
    elif signal_type == 'cos':
        y = np.cos(x)
    return x, y
