import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(XTrain, yTrain), (XTest, yTest) = mnist.load_data()

fig = plt.figure()
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    ax.imshow(XTrain[i], cmap='gray', interpolation='none')
    ax.set_title(yTrain[i])
plt.tight_layout()  
    
XTrain = XTrain.reshape(60000, 784)
XTest = XTest.reshape(10000, 784)
XTrain = XTrain/255
XTest  = XTest/255

YTrain = to_categorical(yTrain, 10)
YTest  = to_categorical(yTest, 10)

myANN = Sequential()
myANN.add(Dense(80,input_dim=784,activation='relu'))
myANN.add(Dense(40,activation='relu'))
myANN.add(Dense(10,activation='sigmoid'))
myANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
myANN.fit(XTrain, YTrain, batch_size=24, epochs=10, verbose=True)

score = myANN.evaluate(XTest, YTest, verbose=False)
print('Test score:', score[0])
print('Test accuracy:', score[1])

errors = 0 
compare = '314159265359'
fig = plt.figure()
for i in range(1,13):
    filename = 'pi-'+str(i)+'.png'
    im = 1 - plt.imread(filename)
    ax = fig.add_subplot(1,13,i+1)
    ax.imshow(im, cmap='gray', interpolation='none')
    number = im.reshape(1,784)
    out = myANN.predict(number)
    print(np.argmax(out), end = '')
    if i == 1: print(',', end = '')
    if compare[i-1] != str(np.argmax(out)): errors += 1
print()
print('3,14159265359')