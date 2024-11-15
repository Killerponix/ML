import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X = np.hstack( (np.linspace(0,0.45,num=50),np.linspace(0.55,1,num=50)) )
Y = (X > 0.5).astype('float').T
Y[1] = Y[5] =  Y[10] = 1 
Y[80] = Y[70] = Y[60] = 0 

#x = np.arange(-3.0, 3.0, 0.25); y = np.arange(-3.0, 3.0, 0.25)
x = np.arange(-1.0, 3.0, 0.1); y = np.arange(-1.0, 2.0, 0.1)
XPlot, YPlot = np.meshgrid(x, y)
XPlotv = XPlot.flatten(); YPlotv = YPlot.flatten()

myANN = Sequential()
myANN.add(Dense(1,input_dim=1,activation='sigmoid'))    
myANN.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
w = myANN.layers[0].get_weights()

Zloss  = np.zeros_like(YPlotv)
Zacc   = np.zeros_like(YPlotv)            
for i in range(XPlotv.shape[0]):
    w[0][0][0]= XPlotv[i]
    w[1][0]   = YPlotv[i]
    myANN.layers[0].set_weights(w)
    Zloss[i] , Zacc[i] = myANN.evaluate(X,Y,verbose=False)
    
Zloss = Zloss.reshape(XPlot.shape)
Zacc  = Zacc.reshape(XPlot.shape)
 
fig = plt.figure()
fig.suptitle("Binary Crossentropy", fontsize="x-large")
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(XPlot, YPlot, Zloss, cmap=cm.Greys)
ax.set_xlabel('w'); ax.set_ylabel('Bias')
ax.set_zlabel('Loss-Function')
ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(XPlot, YPlot, Zacc, cmap=cm.Greys)
ax.set_xlabel('w'); ax.set_ylabel('Bias')
ax.set_zlabel('Acc')
plt.tight_layout()
plt.show(block=False)



w[0][0][0]= 0.5
w[1][0]   = 1.0
myANN = Sequential()
myANN.add(Dense(1,input_dim=1,activation='sigmoid'))    
myANN.compile(loss='binary_crossentropy', optimizer='SGD',metrics=['accuracy'])
myANN.layers[0].set_weights(w)
xErrorSGD = [w[0][0][0]]
yErrorSGD = [w[1][0]]
zErrorSGD = [ myANN.evaluate(X,Y,verbose=False)[0] ]
for i in range(800):
    print(i)
    myANN.fit(X,Y,epochs=1)
    w = myANN.layers[0].get_weights()
    xErrorSGD.append(w[0][0][0])
    yErrorSGD.append(w[1][0])
    zErrorSGD.append(myANN.evaluate(X,Y,verbose=False)[0])
    
w[0][0][0]= 0.5
w[1][0]   = 1.0
myANN = Sequential()
myANN.add(Dense(1,input_dim=1,activation='sigmoid'))    
myANN.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
myANN.layers[0].set_weights(w)
xErrorAdam = [w[0][0][0]]
yErrorAdam = [w[1][0]]
zErrorAdam = [ myANN.evaluate(X,Y,verbose=False)[0] ]
for i in range(800):
    print(i)
    myANN.fit(X,Y,epochs=1)
    w = myANN.layers[0].get_weights()
    xErrorAdam.append(w[0][0][0])
    yErrorAdam.append(w[1][0])
    zErrorAdam.append(myANN.evaluate(X,Y,verbose=False)[0])
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_surface(XPlot, YPlot, Zloss, cmap=cm.Greys, alpha=0.6)
ax.plot(xErrorAdam,yErrorAdam,zErrorAdam, 'bo',linewidth=2,markersize=6, label='Adam')
ax.plot(xErrorSGD,yErrorSGD,zErrorSGD, 'r+',linewidth=2,markersize=6, label='SGD')
ax.set_xlabel('w'); ax.set_ylabel('Bias')
ax.set_zlabel('Loss-Function')
plt.legend()
plt.show(block=False)

print(myANN.evaluate(X,Y,verbose=False))