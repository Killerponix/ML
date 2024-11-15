import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

np.random.seed(42)

X = np.random.rand(1200,2)
groupeA = 0.4*np.sin(2*np.pi*X[:,0] + X[:,0]**2) +0.55 > X[:,1]
groupeB = 0.4*np.sin(2*np.pi*X[:,0] + X[:,0]**2) +0.55 <= X[:,1]
Y = np.zeros(X.shape[0])
Y[groupeA] = 1
Y[groupeB] = -1

groupeE = np.abs(0.4*np.sin(2*np.pi*X[:,0] + X[:,0]**2) +0.55 - X[:,1])< 0.2
index = np.flatnonzero(groupeE)
flip = np.random.rand(index.shape[0]) < 1/4
Y[index[flip]] = (-1)*Y[index[flip]]
groupeAdata = Y>0
groupeBdata = Y<0
XTrain = X 
YTrain = np.zeros( (X.shape[0],2) )
YTrain[groupeAdata,0] = 1
YTrain[groupeBdata,1] = 1

t = np.linspace(0,1,200)
b = 0.4*np.sin(2*np.pi*t + t**2) +0.55
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X[groupeAdata,0],X[groupeAdata,1],marker='+',c='k')
ax.scatter(X[groupeBdata,0],X[groupeBdata,1],marker='*',c='gray')
ax.plot(t,b,'r--',lw=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Trainingsdaten')

myANN = Sequential()
myANN.add(Dense(256,input_dim=2,kernel_initializer='normal',activation='relu'))
myANN.add(Dense(256,kernel_initializer='random_uniform',activation='relu'))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu'))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu'))
myANN.add(Dense(2,kernel_initializer='normal',activation='sigmoid'))
myANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
myANN.fit(XTrain,YTrain, epochs=500,batch_size=20)

XTest = np.random.rand(20000,2)
yp = myANN.predict(XTest)
groupeAp = yp[:,0] > yp[:,1]
groupeBp = yp[:,1] > yp[:,0]  
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(XTest[groupeAp,0],XTest[groupeAp,1],marker='+',c='k')
ax.scatter(XTest[groupeBp,0],XTest[groupeBp,1],marker='*',c='gray')
ax.plot(t,b,'r--',lw=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Ohne Regularisierung')

myANN = Sequential()
myANN.add(Dense(256,input_dim=2,kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(256,kernel_initializer='random_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
myANN.add(Dense(2,kernel_initializer='normal',activation='sigmoid'))
myANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
myANN.fit(XTrain,YTrain, epochs=500,batch_size=20)

XTest = np.random.rand(20000,2)
yp = myANN.predict(XTest)
groupeAp = yp[:,0] > yp[:,1]
groupeBp = yp[:,1] > yp[:,0]  

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(XTest[groupeAp,0],XTest[groupeAp,1],marker='+',c='k')
ax.scatter(XTest[groupeBp,0],XTest[groupeBp,1],marker='*',c='gray')
ax.plot(t,b,'r--',lw=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('$L_2$-Regularisierung')

from keras.layers import Dropout
myANN = Sequential()
myANN.add(Dense(256,input_dim=2,kernel_initializer='normal',activation='relu'))
myANN.add(Dropout(0.5))
myANN.add(Dense(256,kernel_initializer='random_uniform',activation='relu'))
myANN.add(Dropout(0.5))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu'))
myANN.add(Dropout(0.5))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu'))
myANN.add(Dense(2,kernel_initializer='normal',activation='sigmoid'))
myANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
myANN.fit(XTrain,YTrain, epochs=500,batch_size=20)

XTest = np.random.rand(20000,2)

yp = myANN.predict(XTest)

groupeAp = yp[:,0] > yp[:,1]
groupeBp = yp[:,1] > yp[:,0]  

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(XTest[groupeAp,0],XTest[groupeAp,1],marker='+',c='k')
ax.scatter(XTest[groupeBp,0],XTest[groupeBp,1],marker='*',c='gray')
ax.plot(t,b,'r--',lw=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Dropout-Regularisierung')


myANN = Sequential()
myANN.add(Dense(256,input_dim=2,kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l1(0.00001)))
myANN.add(Dense(256,kernel_initializer='random_uniform',activation='relu',kernel_regularizer=regularizers.l1(0.00001)))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu',kernel_regularizer=regularizers.l1(0.00001)))
myANN.add(Dense(128,kernel_initializer='random_uniform',activation='relu',kernel_regularizer=regularizers.l1(0.00001)))
myANN.add(Dense(2,kernel_initializer='normal',activation='sigmoid'))
myANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
myANN.fit(XTrain,YTrain, epochs=500,batch_size=20)

XTest = np.random.rand(20000,2)

yp = myANN.predict(XTest)

groupeAp = yp[:,0] > yp[:,1]
groupeBp = yp[:,1] > yp[:,0]  

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(XTest[groupeAp,0],XTest[groupeAp,1],marker='+',c='k')
ax.scatter(XTest[groupeBp,0],XTest[groupeBp,1],marker='*',c='gray')
ax.plot(t,b,'r--',lw=2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('$L_1$-Regularisierung')