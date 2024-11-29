import numpy as np
from matplotlib import pyplot as plt

from CARTClassificationTree import BinaryDecisionTree

feature_ids = [0, 1, 7]
x_decimals = 5
threshold = 0.1
min_leafnode_size = 3

trainset = np.loadtxt("Trainingsset.csv", delimiter=',')
trainset = trainset[:, feature_ids]
x_train = trainset[:, 1:]
y_train = trainset[:, 0]
testset = np.loadtxt("Testset.csv", delimiter=',')
testset = testset[:, feature_ids]
x_test = testset[:, 1:]
y_test = testset[:, 0]

cartclassificator = BinaryDecisionTree(x_decimals=x_decimals, threshold=threshold, min_leaf_node_size=min_leafnode_size)
cartclassificator.fit(x_train, y_train)
y_pred = cartclassificator.predict(x_test)
error_count = np.count_nonzero(y_test - y_pred)
errors = (y_test != y_pred)
print(f"Fehler: {error_count}")

XX, YY = np.mgrid[x_train[:, 0].min():x_train[:, 0].max():0.005, x_train[:, 1].min():x_train[:, 1].max():0.005]
X = np.array([XX.ravel(), YY.ravel()]).T
Z = cartclassificator.predict(X).reshape(XX.shape)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.pcolormesh(XX, YY, Z,cmap=plt.cm.Pastel1)
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Set1)
ax.scatter(x_test[errors, 0], x_test[errors, 1], c="yellow", marker='x', s=100)
ax.set_xlabel("Alcohol")
ax.set_ylabel("Flavanoids")
fig.savefig("Aufgabe2.png", dpi=300)
plt.show()
