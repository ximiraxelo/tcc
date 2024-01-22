import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

x = np.arange(10)
x_window = sliding_window_view(x, window_shape=5)

print(x)
print(x_window)
