import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from sklearn import linear_model, datasets
sy.init_printing()


iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

logreg = linear_model.LogisticRegression()
logreg.fit(X, Y)
print logreg
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()