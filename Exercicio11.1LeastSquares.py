# Least squares solving
# y = m x + b
# y0 = m x0 + b
# y1 = m x1 + b
# y2 = m x2 + b
# y3 = m x3 + b
# y4 = m x4 + b
# equivalent to A p = y
# with
# A = [x0 1
#      x1 1
#      x2 1
#      x3 1
#      x4 1]
#  p = [m
#       b]

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

m = 0.5
b = 1

N = 100
x = np.array(range(N))

rng = default_rng(1)

noise_y = rng.standard_normal(N) * 10
y_gt = m * x + b
y = y_gt + noise_y

A = np.array([x, np.ones(N)]).T
solution_gt = np.linalg.lstsq(a=A, b=y_gt)
solution = np.linalg.lstsq(a=A, b=y)

m_est_gt, b_est_gt = solution_gt[0]
m_est, b_est = solution[0]

y_est = x * m_est + b_est
y_est_gt = x * m_est_gt + b_est_gt

plt.plot(x, y_gt, 'k', label="Original Data")
plt.plot(x, y, 'rx', label="Original Data + Noise")
plt.plot(x, y_est_gt, 'b', label="Fitted Model with Original Data")
plt.plot(x, y_est, 'g', label="Fitted Model with noise")
plt.legend()
plt.show()
