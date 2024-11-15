import numpy as np
import matplotlib.pyplot as plt
from CARTDecisionTree import bDecisionTree

dataset = np.loadtxt("iris.csv", delimiter=",")
X = dataset[:,0:4]
Y = dataset[:,4]
skale = False #*\label{code:cart:iris:0}
if skale: X[:,0] = 100*X[:,0]    #*\label{code:cart:iris:1}

Sigma = np.cov(X.T)
(lamb, W) = np.linalg.eig(Sigma)
eigenVar = np.sort(lamb)[::-1]
sumEig = np.sum(lamb)
eigenVar = eigenVar/sumEig
cumVar= np.cumsum(eigenVar)
plt.figure()
plt.bar(range(1,len(eigenVar)+1),eigenVar, alpha=0.25, align='center', 
        label='Varianzanteil', color='gray')
plt.step(range(1,len(eigenVar)+1),cumVar, where='mid', 
        label='Kumulativer Varianzanteil', c='k')
plt.xlabel('Hauptkomponenten'); plt.ylabel('Prozentualer Anteil')
plt.legend()

eigenVarIndex = np.argsort(lamb)[::-1]
WP = W[:,eigenVarIndex[0:2]]
XProj = ( WP.T@X.T ).T

np.random.seed(42)
MainSet = np.arange(0,dataset.shape[0])
Trainingsset = np.random.choice(dataset.shape[0], 120, replace=False)
Testset = np.delete(MainSet,Trainingsset)
Testset = np.delete(MainSet,Trainingsset)
XTrain = XProj[Trainingsset,:]
yTrain = Y[Trainingsset]
XTest = XProj[Testset,:]
yTest = Y[Testset]

fullTree = bDecisionTree(minLeafNodeSize=5)
fullTree.fit(XTrain,yTrain)
yP = fullTree.predict(XTest)
print(yP - yTest, '\n', WP.T)

plt.figure()
plt.scatter(XProj[0:50,0],   XProj[0:50,1],c='red',s=60,alpha=0.6)
plt.scatter(XProj[50:100,0], XProj[50:100,1],c='green',marker='^',s=60,alpha=0.6)
plt.scatter(XProj[100:150,0],XProj[100:150,1],c='blue',marker='*',s=80,alpha=0.6)
plt.xlabel('1. Hauptkomponente'); plt.ylabel('1. Hauptkomponente')
plt.grid(True,linestyle='-',color='0.75')