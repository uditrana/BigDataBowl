import numpy as np

dtype = np.float32
# model constants
T = np.linspace(0.1, 4, 40, dtype=dtype)
x = np.linspace(0.5, 119.5, 120, dtype=dtype)
y = np.linspace(-0.5, 53.5, 55, dtype=dtype)
y[0] = -0.2
xx, yy = np.meshgrid(x, y)
field_locs = np.stack((xx, yy)).reshape(2, -1).T  # (F, 2)
tot_pass_cnt = len(field_locs[:, 1])*len(T)
