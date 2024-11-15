import numpy as np
import matplotlib.pyplot as plt
from fullMLP import MLPNet

def regressionPlot(x,y,network,setName=''):
    yp = network.predict(x)         #*\label{code:regPlot:0}
    A = np.ones( (x.shape[0],2) )   #*\label{code:regPlot:1}
    A[:,0] = yp.reshape(x.shape[0]) #*\label{code:regPlot:2} 
    m, c= np.linalg.lstsq(A,y)[0]   #*\label{code:regPlot:3}
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    myTitle = '%s: %f * x + %f ' %(setName, m,c)
    ax.set_title(myTitle)
    ax.scatter(yp,y, marker='+', color=(0.5,0.5,0.5))
    ax.set_xlabel('ANN Output')
    ax.set_ylabel('Data Set')
    alpha = min(yp.min(),y.min())
    omega = max(yp.max(),y.max())
    xPlot = np.linspace(alpha,omega,10)
    ax.plot(xPlot,xPlot,'k')
    ax.plot(xPlot,xPlot*m+c,'r--')
    ax.set_xlim([alpha,omega])
    ax.set_ylim([alpha,omega])

np.random.seed(42)

X = np.loadtxt("BostonFeature.csv", delimiter=",")
Y = np.loadtxt("BostonTarget.csv", delimiter=",")

yMin = Y.min(axis=0); yMax = Y.max(axis=0) 
Y = (Y - yMin) / (yMax - yMin) 
TrainSet     = np.random.choice(X.shape[0],int(X.shape[0]*0.80), replace=False)
XTrain       = X[TrainSet,:]; YTrain       = Y[TrainSet]
TestSet      = np.delete(np.arange(0, len(Y) ), TrainSet) 
XTest        = X[TestSet,:]; YTest        = Y[TestSet]

myPredict = MLPNet(hiddenlayer=(10,10))
myPredict.fit(XTrain,YTrain, eta=0.5, maxIter=1000, XT=XTest , YT=YTest)

regressionPlot(XTest,YTest,myPredict, setName='Testset')
regressionPlot(X,Y,myPredict, setName='All Data')
regressionPlot(XTrain[myPredict.TrainSet,:],YTrain[myPredict.TrainSet],myPredict, setName='Trainingset')
regressionPlot(XTrain[myPredict.ValSet,:],YTrain[myPredict.ValSet],myPredict, setName='Validationset')

ypTest = myPredict.predict(XTest)    
print('Mittlerer Fehler %0.2f' % (np.mean(np.abs(ypTest - YTest[:,None]))))

yp = myPredict.predict(XTrain)
yp = yp.reshape(yp.shape[0])
error = (yMax - yMin)*(yp - YTrain)
print(np.mean(np.abs(error)))

import matplotlib.pyplot as plt
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
epochen = np.arange(len(myPredict.errorVal))
ax.plot(epochen, np.array(myPredict.errorVal), 'r-.' , label='Validierung')  
ax.plot(epochen, np.array(myPredict.errorTest), 'k--', label='Test')   
ax.plot(epochen, np.array(myPredict.errorTrain), 'k:', label='Training' )  
ax.legend()
ax.set_xlabel('Lernzyklus')
ax.set_ylabel('Durchschnittlicher Fehler')

yp = myPredict.predict(XTest)
yp = yp.reshape(yp.shape[0])
errorT = (yMax - yMin)*(yp - YTest)
print(np.mean(np.abs(errorT)))

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('Verteilung der Abweichungen auf der Trainingsmenge')
ax.hist(error,color='gray')
ax.set_xlabel('Abweichung in Tausenden')
ax.set_ylabel('Anzahl')
ax = fig.add_subplot(1,2,2)
ax.set_title('Verteilung der Abweichungen auf der Testmenge')
ax.hist(errorT,color='gray')
ax.set_xlabel('Abweichung in Tausenden')
ax.set_ylabel('Anzahl')
