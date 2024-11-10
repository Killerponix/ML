import numpy as np
import sys
sys.path.append("C:/Users/Simon/Desktop/Studium/5.Semester/Maschinelles Lernen/ML/Programme/Model")
from Programme.Model.CARTDecisionTree import *
from Programme.Model.util.accuracy import *

train = np.loadtxt(delimiter=",",fname="Trainingsset.csv")
test = np.loadtxt(delimiter=",",fname="Testset.csv")

y_train = train[:, 0]       # Erste Spalte ist y für das Trainingsset
X_train = train[:, 1:]      # Restliche Spalten sind X für das Trainingsset

y_test = test[:, 0]         # Erste Spalte ist y für das Testset
X_test = test[:, 1:]        # Restliche Spalten sind X für das Testset

CC = bDecisionTree()
CC.fit(X_train,y_train)
y_predict = CC.predict(X_test)
print(y_predict - y_test)
accuracy(y_test,y_predict)