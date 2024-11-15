import numpy as np
from tensorflow.keras.datasets import cifar10
	
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data() 
noOfClasses = 10
im = []
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(noOfClasses):
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    frist = np.flatnonzero(yTrain == i)[0] 
    im.append(xTrain[frist,:,:,:])
    ax.set_title(i)
    ax.imshow(im[i])
plt.show()

from tensorflow.keras.utils import to_categorical
YTrain = to_categorical(yTrain, noOfClasses)
YTest  = to_categorical(yTest, noOfClasses)
XTrain = xTrain/255.0
XTest  = xTest/255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

l2Reg = 0.001

CNN = Sequential()
CNN.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg),input_shape=(32,32,3)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Flatten())
CNN.add(layers.Dense(512,activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.Dense(256,activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.Dense(10,activation='softmax'))
CNN.summary()
CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
CNN.fit(XTrain,YTrain,epochs=20,batch_size=64)
scores = CNN.evaluate(XTest,YTest,batch_size=64)
print("Accuracy: %.2f%%" % (scores[1]*100))

yPred = CNN.predict(XTest)
choise = np.argmax(yPred, axis=1)

konfusionMatrix = np.zeros((noOfClasses,noOfClasses))
for i in range(noOfClasses):
    index = np.flatnonzero(yTest == i)
    for j in range(10):
        index2 = np.flatnonzero(choise[index] == j)
        konfusionMatrix[i,j] = len(index2)
print(konfusionMatrix)

from keras import backend as K

def outputMoasik(imSingle, outLayer):
    pic = imSingle[np.newaxis,...]
    
    outputSingleLayer = K.function([CNN.layers[0].input],[CNN.layers[outLayer].output])
    picFilter = outputSingleLayer([pic])[0]
    
    gridy = 8
    gridx = 4 if outLayer < 4 else 8
    size = picFilter[0,:,:,0].shape[0] 
    mosaik = np.zeros( (gridx*size,gridy*size))
    
    for l in range(0,picFilter.shape[3]):
        x = int(np.floor(l / gridy))
        y = l%gridy
        mosaik[x*size:(x+1)*size,y*size:(y+1)*size] = picFilter[0,:,:,l]
    plt.figure()
    plt.imshow(mosaik,cmap='binary')
    plt.show()

outputMoasik(im[6], 2)
