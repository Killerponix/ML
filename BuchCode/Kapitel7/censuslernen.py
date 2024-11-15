import numpy as np
from simpleMLP import simpleMLP

np.random.seed(42)

dataset = np.loadtxt("censusGerman.csv", delimiter=",")
XTrain = dataset[:,0][:,None]
YTrain = dataset[:,1]
yMin = YTrain.min(axis=0); yMax = YTrain.max(axis=0) 
YTrain = 2*(YTrain - yMin) / (yMax - yMin) - 1

XTest = np.array([1950.0, 2014.0, 2015.0])[:,None]
YTest = np.array([49.842, 80.896, 82.175 ])

myPredict = simpleMLP(hiddenlayer=(16,16))
myPredict.fit(XTrain,YTrain,maxIter=400, eta=0.25)
yp = myPredict.predict(XTest)
yp = (yMax - yMin)/2* (yp +1) + yMin
fehler = np.sum(np.abs(YTest - yp.T))/len(YTest) 
print('Mittler Fehler fuer die Extrapolation : %f' % (fehler) )
yp = myPredict.predict(XTrain)
yp = (yMax - yMin)/2* (yp +1) + yMin
fehler = np.sum(np.abs(dataset[:,1]-yp.T))/len(YTrain)
print('Mittler Fehler fuer die Regression im Inneren : %f' % (fehler) )

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
ax = fig1.add_subplot(1,1,1)
epochen = np.arange(len(myPredict.error))
xplot = np.arange(1945,2031)[:,None]
yp = myPredict.predict(xplot)
yp = (yMax - yMin)/2* (yp +1) + yMin
ax.plot(xplot , yp, 'k--' ) 
ax.plot(XTrain , dataset[:,1], 'o', markeredgewidth=1,markeredgecolor=(0.5, 0.5, 0.5),markerfacecolor='None' ) 
ax.plot(XTest , YTest, '+', markeredgewidth=1,markeredgecolor='r',markerfacecolor='None' ) 
ax.set_xlabel('Jahr')
ax.set_ylabel('Bevölkerung')


fig2 = plt.figure(2)
ax = fig2.add_subplot(1,1,1)
epochen = np.arange(len(myPredict.error))
ax.plot(epochen, np.array(myPredict.error), 'k' )  
ax.set_xlabel('Lernzyklus')
ax.set_ylabel('Durchschnittlicher Fehler')
