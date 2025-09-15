import matplotlib.pyplot as plt


def visualize_signal(x, y, title):
    """
    Visualize a signal.

    Args:
        x (np.ndarray): The x values of the signal.
        y (np.ndarray): The y values of the signal.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(4, 2))
    plt.plot(x, y, 'm--')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
