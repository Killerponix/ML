import tensorflow as tf 

myDevices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= myDevices, device_type='CPU')

import numpy as np
import matplotlib.pyplot as plt 
from timeit import default_timer as timer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

myActivationList = ['sigmoid','tanh', 'relu', 'elu', 'softplus', 'hard_sigmoid'  ]
myInitializerList = [ 'normal' , 'random_uniform' ,'TruncatedNormal', 'glorot_normal' , 
                     'glorot_uniform', 'he_normal', 'he_uniform']

X = np.linspace(-0.75,1.25,10000)
Y = np.abs(X)

lastError = []; lastTime = []
showPlot  = True
batchNorm = True
noOfRuns = 25
howDeep  = 10

if showPlot:
    plt.figure()
    plt.scatter(X,Y)
    
for myAct in myActivationList: 
    lastError.append([]); lastTime.append([])
    for myInit in myInitializerList:
        minerror = np.Inf
        for c in range(noOfRuns):
            myANN = Sequential()
            myANN.add(Dense(4,input_dim=1,kernel_initializer = myInit, activation = myAct))
            if batchNorm: myANN.add(BatchNormalization())
            for _ in range(1,howDeep):
                myANN.add(Dense(4,kernel_initializer = myInit,activation = myAct))
                if batchNorm: myANN.add(BatchNormalization())
            myANN.add(Dense(1,kernel_initializer = myInit,activation = 'linear'))
            myANN.compile(loss='mse', optimizer='adam')
            start = timer()
            for _ in range(20):
                history = myANN.fit(X,Y, epochs=10, verbose=False)
                if history.history['loss'][-1] < 0.9*10**-2: break
            lastTime[-1].append(timer() - start)
            lastError[-1].append(myANN.evaluate(X,Y,verbose=False))
            if showPlot and minerror > lastError[-1][-1]:
                yPlot = myANN.predict(X)
                minerror = lastError[-1][-1]
        if showPlot:
            theTitle = myAct + ' ' + myInit
            if minerror < 10**-2: plt.plot(X,yPlot, label=theTitle, linewidth=2)
            else: plt.plot(X,yPlot,':', label=theTitle)

if showPlot: plt.legend()
print(np.mean(np.array(lastTime),axis=1))
print(np.mean(np.array(lastError),axis=1))
np.savetxt('errors'+str(howDeep)+'.csv', np.array(lastError))
np.savetxt('times'+str(howDeep)+'.csv', np.array(lastTime)) 

A = np.array(lastError)
success = np.zeros( (len(myActivationList), len(myInitializerList) ) )
for i in range(len(myActivationList)): 
    for j in range(len(myInitializerList)):
        success[i,j] = np.sum(A[i,j*noOfRuns:(j+1)*noOfRuns]<0.9*10**-2)
np.savetxt('success'+str(howDeep)+'.csv', np.array(success))     