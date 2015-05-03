import numpy as np
from sympy import *
from sklearn import datasets
import math
from IPython.display import display
init_printing()

def make_data():
    iris = datasets.load_iris()
    features = iris.data
    target = iris.target

    training_features_class0 = features[0:40]
    training_features_class1 = features[50:90]
    training_target_class0 = target[0:40]
    training_target_class1 = target[50:90]

    testing_features_class0 = features[41:50]
    testing_features_class1 = features[91:100]
    testing_target_class0 = target[41:50]
    testing_target_class1 = target[91:100]

    training_features = np.append(training_features_class0, training_features_class1, axis=0)
    training_target = np.append(training_target_class0, training_target_class1, axis=0)
    testing_features = np.append(testing_features_class0, testing_features_class1, axis=0)
    testing_target = np.append(testing_target_class0, testing_target_class1, axis=0)

    training_target = training_target.reshape((80, 1))
    testing_target = testing_target.reshape((18, 1))

    return training_features, training_target, testing_features, testing_target

def get_probabilities(X, b):
    return 1 / (1.0 + math.e ** (-1 * np.dot(X, b)))

def gradient(X, Y, b):
    p = get_probabilities(X, b)
    return np.dot(-1.0 * X.transpose(), (Y - p))

Xt, Yt, Xf, Yf = make_data()
b0 = np.array([1, 1, 1, 1]).reshape((4, 1))
step = 0.1
for i in range(0, 10000):
    b0 = b0 - step*gradient(Xt, Yt, b0)
print b0

test = get_probabilities(Xf, b0)
output = np.append(test, Yf, axis=1)
print output


'''
x = symbols("x")
expr = sin(x)
display(expr)
f = expr.evalf(subs={x:10})
'''

