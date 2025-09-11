import numpy as np
from signal_ops.signal_utils import make_signal
from visualizations.vis_utils import visualize_signal

x, y = make_signal('sin', (0, 2 * np.pi))
visualize_signal(x, y, 'Sine Wave')
