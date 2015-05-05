import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

sy.init_printing()

x = sy.symbols('x')
sigmoid = 1 / (1 + sy.exp(-x))
lam_sig = sy.utilities.lambdify(x, sigmoid, np)

x_axis = np.arange(-6, 6, .01)
y_axis = lam_sig(x_axis)

display(sigmoid)

fig = plt.figure()
sigplot = fig.add_subplot(1, 1, 1)
sigplot.plot(x_axis, y_axis)
sigplot.spines['left'].set_position('center')
sigplot.spines['bottom'].set_position('center')
sigplot.spines['top'].set_color('none')
sigplot.spines['right'].set_color('none')
sigplot.xaxis.set_ticks_position('bottom')
sigplot.yaxis.set_ticks_position('left')
plt.show()