import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)
X = np.loadtxt("BostonFeature.csv", delimiter=",") #*\label{code:lstqBoston:1}
y = np.loadtxt("BostonTarget.csv", delimiter=",")
TrainSet     = np.random.choice(X.shape[0],int(X.shape[0]*0.80), replace=False)
XTrain       = X[TrainSet,:] 
YTrain       = y[TrainSet]
TestSet      = np.delete(np.arange(0, len(y) ), TrainSet) 
XTest        = X[TestSet,:] 
YTest        = y[TestSet] #*\label{code:lstqBoston:2}

A = np.ones( (XTrain.shape[0],14) ) #*\label{code:lstqBoston:3}
A[:,1:14] = XTrain #*\label{code:lstqBoston:9}
maxValue = np.max(A,axis=0) #*\label{code:lstqBoston:10}
A = A/maxValue #*\label{code:lstqBoston:11}
(u, _, Arank, _) = np.linalg.lstsq(A, YTrain) #*\label{code:lstqBoston:4}
r = A@u - YTrain #*\label{code:lstqBoston:12}
print(np.linalg.norm(r)/r.shape[0], np.mean(np.abs(r)), np.max(np.abs(r))) #*\label{code:lstqBoston:5}
print(u) #*\label{code:lstqBoston:6}

B = np.ones( (XTest.shape[0],14) ) #*\label{code:lstqBoston:7}
B[:,1:14] = XTest
B = B /maxValue
yPredit = B@u #*\label{code:lstqBoston:8}
rT = yPredit - YTest
print(np.linalg.norm(rT)/rT.shape[0], np.mean(np.abs(rT)), np.max(np.abs(r)))

fig = plt.figure(1)
ax = fig.add_subplot(1,2,1)
ax.set_title('Verteilung der Abweichungen auf der Trainingsmenge')
ax.hist(r,color='gray')
ax.set_xlabel('Abweichung in Tausenden')
ax.set_ylabel('Anzahl')
ax = fig.add_subplot(1,2,2)
ax.set_title('Verteilung der Abweichungen auf der Testmenge')
ax.hist(rT,color='gray')
ax.set_xlabel('Abweichung in Tausenden')
ax.set_ylabel('Anzahl')