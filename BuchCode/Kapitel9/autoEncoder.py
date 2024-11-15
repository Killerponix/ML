import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist

(XTrain, YTrain), (XTest, YTest) = mnist.load_data()
XTrain = XTrain.reshape(60000, 784)
XTest = XTest.reshape(10000, 784)
XTrain = XTrain/255
XTest  = XTest/255

Layer1 = 196
Layer2 = 98 
zielZahlMerkmale = 24

autoencoder = Sequential()
autoencoder.add(Dense(Layer1,input_dim=784,activation='sigmoid'))
autoencoder.add(Dense(Layer2,activation='relu'))
autoencoder.add(Dense(zielZahlMerkmale,activation='relu'))
autoencoder.add(Dense(Layer2,activation='relu'))
autoencoder.add(Dense(Layer1,activation='relu'))
autoencoder.add(Dense(784,activation='sigmoid'))
autoencoder.compile(loss='mean_squared_error', optimizer='adam')
autoencoder.fit(XTrain, XTrain, epochs=25, verbose=True,validation_data=(XTest, XTest))

encoder = Sequential()
encoder.add(Dense(Layer1,input_dim=784,activation='sigmoid'))
encoder.add(Dense(Layer2,activation='relu'))
encoder.add(Dense(zielZahlMerkmale,activation='relu'))

for i in range(len(encoder.layers)):
    W = autoencoder.layers[i].get_weights()
    encoder.layers[i].set_weights(W)
    
decoder = Sequential()
decoder.add(Dense(Layer2,input_dim=zielZahlMerkmale, activation='relu'))
decoder.add(Dense(Layer1,activation='relu'))
decoder.add(Dense(784,activation='sigmoid'))

for i in range(len(encoder.layers),len(autoencoder.layers)):
    W = autoencoder.layers[i].get_weights()
    decoder.layers[i-len(encoder.layers)].set_weights(W)
    
encodedData = encoder.predict(XTest)
decodedData = decoder.predict(encodedData)

plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(XTest[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decodedData[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

XTrainRed = encoder.predict(XTrain)
XTestRed = encoder.predict(XTest)

def unscaledKNNclassification(xTrain, yTrain, xQuery, k, normOrd=None):
    diff = xTrain - xQuery
    dist = np.linalg.norm(diff,axis=1, ord=normOrd)
    knearest = np.argpartition(dist,k)[0:k]
    (classification, counts) = np.unique(YTrain[knearest], return_counts=True)
    theChoosenClass = np.argmax(counts) 
    return(classification[theChoosenClass])


errors = 0
for i in range(len(YTest)):
    myClass = unscaledKNNclassification(XTrain, YTrain, XTest[i,:], 3)
    if myClass != YTest[i]:
        errors = errors +1
        print('%d wurde als %d statt %d klassifiziert' % (i,myClass,YTest[i]))
