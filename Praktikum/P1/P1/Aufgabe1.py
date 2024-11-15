import numpy as np
import sys

sys.path.append("C:/Users/Simon/Desktop/Studium/5.Semester/Maschinelles Lernen/ML/Programme/Model")
from Programme.Model.CARTDecisionTree import *
from Programme.Model.util.accuracy import *
import matplotlib.pyplot as plt


train = np.loadtxt(delimiter=",",fname="Trainingsset.csv")
test = np.loadtxt(delimiter=",",fname="Testset.csv")
data = np.loadtxt(delimiter=",",fname="AllData.csv")

y_train = train[:, 0]       # Erste Spalte ist y für das Trainingsset
X_train = train[:, 1:]      # Restliche Spalten sind X für das Trainingsset

y_test = test[:, 0]         # Erste Spalte ist y für das Testset
X_test = test[:, 1:]        # Restliche Spalten sind X für das Testset

CC = bDecisionTree(xDecimals=5,threshold=0.1,minLeafNodeSize=3)
CC.fit(X_train,y_train)
y_predict = CC.predict(X_test)
print(y_predict - y_test)
accuracy(y_test,y_predict)

fig = plt.figure(1)
ax = fig.add_subplot(2,2,1)
ax.scatter(data[:,8],data[:,13])
ax.set_xlabel('Non flavanoid phenols')
ax.set_ylabel('Proline')
ax.grid(True)

ax = fig.add_subplot(2,2,2)
ax.scatter(data[:,7],data[:,10])
ax.set_xlabel('Flavanoids')
ax.set_ylabel('Color intensity')
ax.grid(True)

ax = fig.add_subplot(2,2,3)
ax.scatter(data[:,1],data[:,7])
ax.set_xlabel('Alcohol')
ax.set_ylabel('Flavanoids')
ax.grid(True)

ax = fig.add_subplot(2,2,4)
ax.scatter(data[:,1],data[:,10])
ax.set_xlabel('Alcohol')
ax.set_ylabel('Color intensity')
ax.grid(True)

plt.tight_layout()
plt.show(block=False)
#fig.show(True, block=False)
fig.savefig('Aufgabe_1.pdf', bbox_inches='tight')



