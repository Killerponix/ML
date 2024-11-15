import numpy as np
from simpleMLP import simpleMLP

np.random.seed(42)
X = np.random.rand(1250,2)
Y = np.zeros(1250)
index = (X[:,0] - 0.5)**2 + (X[:,1] - 0.5)**2 < 0.2**2
Y[index] = 1

TrainSet     = np.random.choice(X.shape[0],int(X.shape[0]*0.80), replace=False)
XTrain       = X[TrainSet,:]
YTrain       = Y[TrainSet]
falsePositive = np.random.choice(len(TrainSet),int(len(TrainSet)*0.15), replace=False) #*\label{code:kreislernendreck:1}   
YTrain[falsePositive] = 1 #*\label{code:kreislernendreck:2}
TestSet      = np.delete(np.arange(0, len(Y) ), TrainSet) 
XTest        = X[TestSet,:]
YTest        = Y[TestSet]

myPredict = simpleMLP(hiddenlayer=(120,120))
myPredict.fit(XTrain,YTrain,maxIter=600)
yp = myPredict.predict(XTest)
diff = np.abs(np.round(yp).T - YTest)
fp = np.sum(diff)/len(TestSet)*100 
print('richtig klassifiziert %0.1f%%' % (100-fp))
print('falsch klassifiziert %0.1f%%' % (fp))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig1 = plt.figure(1)
ax = fig1.add_subplot(1,1,1)
ax.scatter(XTrain[:,0],XTrain[:,1],c =YTrain, marker='o', edgecolors='k', cmap=cm.Greys)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')

fig2 = plt.figure(2)
ax = fig2.add_subplot(1,1,1)
epochen = np.arange(len(myPredict.error))
ax.plot(epochen, np.array(myPredict.error), 'k' )  
ax.set_xlabel('Lernzyklus')
ax.set_ylabel('Durchschnittlicher Fehler')

fig1 = plt.figure(3)
ax = fig1.add_subplot(1,1,1)
ax.scatter(XTest[:,0],XTest[:,1],c =np.round(yp).reshape(yp.shape[0]), marker='o', edgecolors='k', cmap=cm.Greys)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')