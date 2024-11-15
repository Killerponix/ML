import numpy as np
import sys

sys.path.append("C:/Users/Simon/Desktop/Studium/5.Semester/Maschinelles Lernen/ML/Programme/Model")
from Programme.Model.RandomForestDecision import *
from Programme.Model.util.accuracy import *
import matplotlib.pyplot as plt


train = np.loadtxt(delimiter=",",fname="Trainingsset.csv")
test = np.loadtxt(delimiter=",",fname="Testset.csv")
data = np.loadtxt(delimiter=",",fname="AllData.csv")

y_train = train[:, 0]       # Erste Spalte ist y für das Trainingsset
X_train = train[:, 1:]      # Restliche Spalten sind X für das Trainingsset

y_test = test[:, 0]         # Erste Spalte ist y für das Testset
X_test = test[:, 1:]        # Restliche Spalten sind X für das Testset

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
errors = []

for i in range(1,50):
    rf = randomForestDecision()
    rf.fit(X_train,y_train)
    y_predict = rf.predict(X_test)
    accuracy(y_test,y_predict)
    error = (np.abs(np.sum(y_predict-y_test)))
    errors.append(error)

ax.plot(range(1,50),errors,linestyle='-')
ax.set_xlabel("Bäume")
ax.set_ylabel("Fehler")
fig.show(True)
