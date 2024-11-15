import numpy as np

np.random.seed(45)

fString = open("diagnosis.data","r")
fFloat  = open("diagnosis.csv","w")
for line in fString:
    line = line.replace(",", ".")
    line = line.replace("\t", ",")
    line = line.replace("yes", "1")
    line = line.replace("no", "0")
    line = line.replace("\r\n", "\n")
    fFloat.write(line)
fString.close()
fFloat.close()
dataset = np.loadtxt("diagnosis.csv", delimiter=",")

X = dataset[:,1:6]
Y = dataset[:,6]
allData     = np.arange(0,X.shape[0])
iTesting    = np.random.choice(X.shape[0],int(X.shape[0]*0.2),replace=False)
iTraining   = np.delete(allData,iTesting) 
dataRecords = len(iTraining)
XTrain = X[iTraining,:]
YTrain = Y[iTraining]

PXI = np.zeros( (2,XTrain.shape[1],2) )
PI  = np.zeros(2)
for k in range(X.shape[1]):
    PXI[1,k,1] = np.sum(np.logical_and(XTrain[:,k],YTrain))
    PXI[1,k,0] = np.sum(np.logical_and(np.logical_not(XTrain[:,k]),YTrain))  
    PXI[0,k,1] = np.sum(np.logical_and(XTrain[:,k],np.logical_not(YTrain)))
    PXI[0,k,0] = np.sum(np.logical_not(np.logical_or(XTrain[:,k],YTrain)))
PI[1] = np.sum(YTrain)
PI[0] = dataRecords - PI[1]

PXI = (PXI + 1/2) / (dataRecords+1)
PI  = PI  / dataRecords
  
def predictNaiveBayesNominal(x):
    P = np.zeros_like(PI)
    allofthem = np.arange(XTrain.shape[1])
    for i in range(len(PI)):
        P[i] = np.prod(PXI[i,allofthem,x])*PI[i]
    denominator = np.sum(P)
    P = P/denominator
    choosenClass = np.argmax(P)
    return choosenClass

XTest = X[iTesting,:]
YTest = Y[iTesting]
correct   = np.zeros(2)
incorrect = np.zeros(2)

for i in range(XTest.shape[0]):
    klasse = predictNaiveBayesNominal(XTest[i,:].astype(int))
    if klasse == YTest[i]:
        correct[klasse] = correct[klasse] +1
    else:
        incorrect[klasse] = incorrect[klasse] +1
        
print("Von %d Testfaellen wurden %d richtig und %d falsch klassifiziert" % (XTest.shape[0],np.sum(correct),np.sum(incorrect) ))

T = dataset[:,0]
trueIndex = np.flatnonzero(YTrain==1)
falseIndex = np.flatnonzero(YTrain==0)
muApproxTrue = np.sum(T[trueIndex])/trueIndex.shape[0]
sgApproxTrue = np.sum( (T[trueIndex]-muApproxTrue)**2 ) / (trueIndex.shape[0] -1)
muApproxFalse = np.sum(T[falseIndex])/falseIndex.shape[0]
sgApproxFalse = np.sum( (T[falseIndex]-muApproxFalse)**2 ) / (falseIndex.shape[0] -1)

def Gausverteilung(x,mu,sigma):
    y = np.exp(-0.5*( (x-mu)/sigma)**2 )/(sigma*np.sqrt(2*np.pi))
    return(y)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(131)
ax.hist(T[:],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur'); 
ax.set_ylabel('Wahrscheinlichkeit')
Tplot = np.arange(33,44,0.05)
ax.plot(Tplot,Gausverteilung(Tplot,muApproxTrue,sgApproxTrue),'k:')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxFalse,sgApproxFalse),'k-.')
ax.set_ylim([0,0.8])
ax.set_title('Alle Trainingsdaten')
ax = fig.add_subplot(132)
ax.hist(T[falseIndex],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxFalse,sgApproxFalse),'k-.')
ax.set_ylim([0,0.8])
ax.set_title('Negative Diagnose')
ax = fig.add_subplot(133)
ax.hist(T[trueIndex],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxTrue,sgApproxTrue),'k:')
ax.set_ylim([0,0.8])
ax.set_title('Positive Diagnose')
plt.tight_layout()
plt.show(block=False)

def predictNaiveBayesMixed(x,T,muTrue,sigmaTrue,muFalse,sigmaFalse):
    P = np.zeros_like(PI)
    allofthem = np.arange(XTrain.shape[1])
    P[0] = np.prod(PXI[0,allofthem,x])*PI[0]
    P[1] = np.prod(PXI[1,allofthem,x])*PI[1]
    P[0] = P[0] * Gausverteilung(T, muFalse,sigmaFalse)
    P[1] = P[1] * Gausverteilung(T, muTrue,sigmaTrue)
    choosenClass = np.argmax(P)
    return choosenClass
 
TTest = T[iTesting]    
def TestNaiveBayesMixed(muTrue,sigmaTrue,muFalse,sigmaFalse):
    correct   = np.zeros(2); incorrect = np.zeros(2)
    for i in range(XTest.shape[0]):
        klasse = predictNaiveBayesMixed(XTest[i,:].astype(int),TTest[i],muTrue,sigmaTrue,muFalse,sigmaFalse)
        if klasse == YTest[i]:
            correct[klasse] = correct[klasse] +1
        else:
            incorrect[klasse] = incorrect[klasse] +1    
    return(correct, incorrect)

(correct, incorrect) =  TestNaiveBayesMixed(muApproxTrue,sgApproxTrue, muApproxFalse, sgApproxFalse)
print("Von %d Testfaellen wurden %d richtig und %d falsch klassifiziert" % (XTest.shape[0],np.sum(correct),np.sum(incorrect) ))    

keineDiagnose = np.logical_not(np.logical_or(dataset[iTraining,7],YTrain))
index = np.flatnonzero(keineDiagnose)
muApprox = np.sum(T[index])/index.shape[0]
sgApprox = np.sum( (T[index]-muApprox)**2 ) / (index.shape[0] -1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(T[index],15,density=1, facecolor='k', alpha=0.5)
ax.set_xlabel('Temperatur'); 
ax.set_ylabel('Wahrscheinlichkeit')
ax.plot(Tplot,Gausverteilung(Tplot,muApprox,sgApprox),'k')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxTrue,sgApproxTrue),'k:')
ax.plot(Tplot,Gausverteilung(Tplot,muApproxFalse,sgApproxFalse),'k-.')
