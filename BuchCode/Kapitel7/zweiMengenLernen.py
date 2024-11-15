import numpy as np
from simpleMLP import simpleMLP

np.random.seed(42)
X = np.random.rand(1250,2)
Y = np.zeros(1250)
index = (X[:,0] - 0.25)**2 + (X[:,1] - 0.25)**2 < 0.2**2
Y[index] = 1
index = (X[:,0] - 0.75)**2 + (X[:,1] - 0.75)**2 < 0.2**2
Y[index] = 2

TrainSet     = np.random.choice(X.shape[0],int(X.shape[0]*0.70), replace=False)
XTrain       = X[TrainSet,:]
YTrain       = Y[TrainSet]
TestSet      = np.delete(np.arange(0, len(Y) ), TrainSet) 
XTest        = X[TestSet,:]
YTest        = Y[TestSet]

myPredict = simpleMLP(hiddenlayer=(32,32))
myPredict.fit(XTrain,YTrain,maxIter=2000)
yp = myPredict.predict(XTest)
diff = np.abs(np.round(yp.T) - YTest).astype(bool)
fp = np.sum(diff)/len(TestSet)*100 
print('richtig klassifiziert %0.1f%%' % (100-fp))
print('falsch klassifiziert %0.1f%%' % (fp))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.close('all')
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
yp = yp.reshape(XTest.shape[0])
fig3 = plt.figure(3)
ax = fig3.add_subplot(1,1,1, projection='3d')
ax.scatter(XTest[:,0],XTest[:,1],yp,alpha=0.6,c =yp, cmap=cm.Greys)
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_zlabel('$y_p$')
