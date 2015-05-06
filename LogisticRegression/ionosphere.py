import numpy as np
import logistic as log

feature_num = 34
rows = 351

dataset = np.genfromtxt('ionosphere2.csv', delimiter=',')
np.random.shuffle(dataset)
features = np.append(np.ones((rows, 1)), dataset, axis=1)
features = dataset[:, 0:feature_num+1]
target = dataset[:, feature_num].reshape((rows, 1))

train_feature, train_target, test_feature, test_target = log.get_data(features, target, 150, rows)

b = np.ones((feature_num+1, 1))
b = log.descent(train_feature, train_target, b)

test = log.sigmoid(test_feature, b)

output = np.append(test, test_target, axis=1)
count = 0.0
correct = 0.0
for i in output:
    if i[0] > 0.5 and i[1] == 1:
        correct += 1
    if i[0] <= 0.5 and i[1] == 0:
        correct += 1
    count += 1

print "{}%".format((correct/count)*100)
