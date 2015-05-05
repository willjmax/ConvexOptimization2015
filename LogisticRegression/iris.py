from logistic import *
from sklearn import datasets

feature_num = 4

def make_data():
    iris = datasets.load_iris()
    features = iris.data
    target = iris.target
    binary_features = features[0:100]
    binary_target = target[0:100].reshape((100, 1))

    data = np.append(binary_features, binary_target, axis=1)
    np.random.shuffle(data)

    binary_features = np.append(np.ones((100, 1)), data[:,0:feature_num], axis=1)
    binary_target = data[:,feature_num].reshape((100, 1))

    return binary_features, binary_target

features, target = make_data()
trf, trt, tef, tet = get_data(features, target, 80, 100)

b = np.ones((feature_num+1, 1))
b = descent(trf, trt, b)

test = sigmoid(tef, b)

output = np.append(test, tet, axis=1)
print output
count = 0.0
correct = 0.0
for i in output:
    if i[0] > 0.5 and i[1] == 1:
        correct += 1
    if i[0] <= 0.5 and i[1] == 0:
        correct += 1
    count += 1

print "{}%".format((correct/count)*100)