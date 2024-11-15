import numpy as np

dataset = np.loadtxt("iris.csv", delimiter=",")
x = dataset[:,0:4]
y = dataset[:,4]
percentTrainingset = 0.8
np.random.seed(42)
TrainSet     = np.random.choice(x.shape[0],int(x.shape[0]*percentTrainingset),replace=False)
XTrain       = x[TrainSet,:]
YTrain       = y[TrainSet]
TestSet      = np.delete(np.arange(0,len(y)), TrainSet) 
XTest        = x[TestSet,:]
YTest        = y[TestSet]

def plainKNNclassification(xTrain, yTrain, xQuery, k, normOrd=None):
    xMin = xTrain.min(axis=0); xMax = xTrain.max(axis=0) #*\label{code:knnscale1}
    xTrain = (xTrain - xMin) / (xMax - xMin)
    xQuery = (xQuery - xMin) / (xMax - xMin) #*\label{code:knnscale2}
    diff = xTrain - xQuery
    dist = np.linalg.norm(diff,axis=1, ord=normOrd)
    knearest = np.argpartition(dist,k)[0:k]
    (classification, counts) = np.unique(yTrain[knearest], return_counts=True)
    theChoosenClass = np.argmax(counts) #*\label{code:knnmehrheit}
    return(classification[theChoosenClass])

errors = 0
for i in range(len(YTest)):
    myClass = plainKNNclassification(XTrain, YTrain, XTest[i,:], 3)
    if myClass != YTest[i]:
        errors = errors +1
        print('%s wurde als %d statt %d klassifiziert' % (str(XTest[i,:]),myClass,YTest[i]))


from twoMoonsProblem import twoMoonsProblem

(XTrain,YTrain) = twoMoonsProblem()

XX, YY = np.mgrid[-1:2:0.01, -1:2:0.01]
X = np.array([XX.ravel(), YY.ravel()]).T
yP = np.zeros(X.shape[0])

for i in range(X.shape[0]):
    yP[i] = plainKNNclassification(XTrain, YTrain, X[i,:], 3,normOrd=np.inf)

indexA = np.flatnonzero(YTrain>0.5)
indexB = np.flatnonzero(YTrain<0.5)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
yP = yP.reshape(XX.shape)
ax.pcolormesh(XX, YY, yP, cmap=plt.cm.Set1)
ax.scatter(XTrain[indexA,0],XTrain[indexA,1],color='white', marker='o')
ax.scatter(XTrain[indexB,0],XTrain[indexB,1],color='black', marker='+')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_xlim([-1,2])
ax.set_ylim([-1,2])
ax.set_title("Klassifikation mit inf-Norm")