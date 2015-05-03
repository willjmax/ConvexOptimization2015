import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from IPython.display import display

def make_data():
    x = 2*np.random.rand(100) - 1
    X = np.array([x, np.ones(100)]).transpose()
    Y = (2*x + np.random.rand(100)).transpose()
    return X, Y, x

def gradient(X, Y, b):
    return np.dot(X.transpose(), np.dot(X, b) - Y)

X, Y, x = make_data()
b = np.array([1, 0])

step = 0.01

for i in range(0,1000):
    b = b - step*gradient(X,Y,b)

fig = plt.figure()
plt.scatter(x, Y)
plot_x = np.arange(-1, 1, .01)

y = b[0]*plot_x + b[1]

plt.plot(plot_x, y, "r")
plt.show()