import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CARTDecisionTree import bDecisionTree

newModel        = True
nurEpistemische = True
PCA             = False

if nurEpistemische:
    b1 = [ 2.5, -3.2, -1.0]
    b2 = [-2.0, +5.5, +3.0]
else:
    b1 = [ 2.0, -2.0, -0.5]
    b2 = [-1.0, +4.0, +2.5]    

DoF = [10000,15000, 20000,30000,40000,60000,80000,100000]
p = []
for sPerC in DoF:
    np.random.seed(42)
    Y = np.ones((2*sPerC))
    Y[0:sPerC] = 0 
    X = np.zeros((2*sPerC,3))    
    X[0:sPerC,0] = 1.2*truncnorm.rvs(-2, +2, size=sPerC) + b1[0]
    X[0:sPerC,1] = 3.0*truncnorm.rvs(-2, +2, size=sPerC) + b1[1]
    X[0:sPerC,2] = 3.0*truncnorm.rvs(-2, +2, size=sPerC) + b1[2]    
    X[sPerC:2*sPerC,0] = 1.0*truncnorm.rvs(-2, +2, size=sPerC) + b2[0]
    X[sPerC:2*sPerC,1] = 1.0*truncnorm.rvs(-2, +2, size=sPerC) + b2[1]
    X[sPerC:2*sPerC,2] = 2.5*truncnorm.rvs(-2, +2, size=sPerC) + b2[2]
    
    if newModel:
        start = int(1.8*sPerC)
        addOn = 2*sPerC - start 
        X[start:2*sPerC,0] = 1.0*truncnorm.rvs(-2, +2, size=addOn) + 1.5
        X[start:2*sPerC,1] = 1.0*truncnorm.rvs(-2, +2, size=addOn) + 6.0
        X[start:2*sPerC,2] = 3.0*truncnorm.rvs(-2, +2, size=addOn) + 3.0 
        
    if PCA: 
        Sigma = np.cov(X.T)
        (lamb, W) = np.linalg.eig(Sigma)
        eigenVarIndex = np.argsort(lamb)[::-1]
        WP = W[:,eigenVarIndex[0:2]]
        X  = ( WP.T@X.T ).T

    MainSet = np.arange(0,X.shape[0])
    Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
    Testset = np.delete(MainSet,Trainingsset)
    XTrain = X[Trainingsset,:]
    YTrain = Y[Trainingsset]
    XTest = X[Testset,:]
    YTest = Y[Testset]
    
    fullTree = bDecisionTree(minLeafNodeSize=7, xDecimals = 2)
    fullTree.fit(XTrain,YTrain)
    yP = fullTree.predict(XTest)
    p.append(np.abs(yP - YTest).sum()/YTest.shape[0]*100)
    print(p[-1],' Prozent der Testmenge wurden fehlklassifiziert')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(DoF,p,c='k',marker='o')
ax.set_xlabel('Anzahl Datensätze')
ax.set_ylabel('Anteil Fehlklassifizierte')
plt.show()
a = 0.1
if not PCA: 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(X[0:sPerC,0],X[0:sPerC,1],X[0:sPerC,2],c='red',s=60,alpha=a)
    ax.scatter(X[sPerC:2*sPerC,0],X[sPerC:2*sPerC,1],X[sPerC:2*sPerC,2],c='black',marker='^',s=60,alpha=a)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[0:sPerC,0],X[0:sPerC,1],c='red',s=60,alpha=a)
    ax.scatter(X[sPerC:2*sPerC,0],X[sPerC:2*sPerC,1],c='black',marker='^',s=60,alpha=a)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
else:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[0:sPerC,0],X[0:sPerC,1],c='red',s=60,alpha=a)
    ax.scatter(X[sPerC:2*sPerC,0],X[sPerC:2*sPerC,1],c='black',marker='^',s=60,alpha=a)
    ax.set_xlabel('Hauptachse 1')
    ax.set_ylabel('Hauptachse 2')
    plt.show()


