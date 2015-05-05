import numpy as np
import math

def get_data(features, target, trainlimit, rows):
    training_features = features[0:trainlimit]
    training_target = target[0:trainlimit].reshape((trainlimit, 1))
    testing_features = features[trainlimit:rows]
    testing_target = target[trainlimit:rows].reshape((rows-trainlimit, 1))
    return training_features, training_target, testing_features, testing_target

def sigmoid(X, b):
    return 1 / (1.0 + math.e ** (-1 * np.dot(X, b)))

def gradient(X, Y, b):
    p = sigmoid(X, b)
    return np.dot(-1.0 * X.transpose(), (Y - p))

def descent(features, targets, b):
    step = 0.01
    for i in range(0, 1000):
        b = b - step*gradient(features, targets, b)
    return b



