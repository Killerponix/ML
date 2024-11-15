import numpy as np
from os import makedirs, path, listdir
from shutil import copyfile

np.random.seed(42)
if not path.isdir('dogs-vs-cats/train/dog'):
    testRatio = 0.25
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
    	labeldirs = ['dogs/', 'cats/']
    	for labldir in labeldirs:
    		newdir = 'dogs-vs-cats/' + subdir + labldir
    		makedirs(newdir, exist_ok=True)
    for file in listdir('dogs-vs-cats/train'):
        src = 'dogs-vs-cats/train/' + file
        if path.isfile(src):
            if np.random.rand() < testRatio: dst_dir = 'test/'
            else: dst_dir = 'train/'
            if file.startswith('cat'):
                dst = 'dogs-vs-cats/' + dst_dir + 'cats/'  + file
                copyfile(src, dst)
            elif file.startswith('dog'):
                dst = 'dogs-vs-cats/' + dst_dir + 'dogs/'  + file
                copyfile(src, dst)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

reducePicDim = 256

try:
    CNN = load_model("dogVScat.h5")
except:
    trainDatagen = ImageDataGenerator(rotation_range=30, rescale=1./255, horizontal_flip=0.1)
    trainGenerator = trainDatagen.flow_from_directory(
        directory=r"./dogs-vs-cats/train/",
        target_size=(reducePicDim, reducePicDim),
        color_mode="rgb", batch_size=64,
        class_mode="categorical", shuffle=True, seed=42)

    CNN = Sequential()
    CNN.add(Conv2D(32,(5,5),activation='relu',input_shape=(reducePicDim,reducePicDim,3)))
    CNN.add(MaxPool2D(pool_size=(3, 3)))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(32,(5,5),activation='relu'))
    CNN.add(MaxPool2D(pool_size=(3, 3)))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(64,(3,3),activation='relu'))
    CNN.add(MaxPool2D(pool_size=(2, 2)))
    CNN.add(BatchNormalization())
    CNN.add(Conv2D(64,(3,3),activation='relu'))
    CNN.add(Flatten())
    CNN.add(Dense(100,activation='relu'))
    CNN.add(Dense(50,activation='relu'))
    CNN.add(Dense(2,activation='softmax'))
    CNN.summary()
    CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    CNN.fit_generator(generator=trainGenerator, epochs=10, verbose=True)
    CNN.save("dogVScat.h5")

testDatagen = ImageDataGenerator(rescale=1./255)
testGenerator = testDatagen.flow_from_directory(
    directory=r"./dogs-vs-cats/test/",
    target_size=(reducePicDim, reducePicDim),
    color_mode="rgb", batch_size=8,
    class_mode="categorical", shuffle=False)
myloss, acc = CNN.evaluate_generator(testGenerator, steps=len(testGenerator), verbose=True)
print('Acc: %.3f' % (acc * 100.0))
testGenerator.reset()
yP = CNN.predict_generator(testGenerator, steps=len(testGenerator), verbose=True)
yPClass = np.argmax(yP,axis=1)
cats = np.sum(testGenerator.classes == 0)
dogs = np.sum(testGenerator.classes == 1)
catsAsDogs = np.sum( np.abs(yPClass[testGenerator.classes == 0] -0) )
dogsAsCats = np.sum( np.abs(yPClass[testGenerator.classes == 1] -1) )
confMatrix = np.array([[(cats-catsAsDogs)/cats, catsAsDogs/cats],
                        [dogsAsCats/dogs, (dogs-dogsAsCats)/dogs]])
print(confMatrix)

import tensorflow as tf    
from tensorflow.keras.models import Model
def heatmap(img, model):
    for l in model.layers: #*\label{code:catsdogs:1}
        if isinstance(l, Conv2D): lastConvLayer = l #*\label{code:catsdogs:2}
    calcFeaturesAndPred = Model([model.input], [lastConvLayer.output, model.output]) #*\label{code:catsdogs:3}
    img = tf.Variable(img); #*\label{code:catsdogs:7}
    with tf.GradientTape() as tape:
        featureMaps, predictions = calcFeaturesAndPred(img[np.newaxis,...])  #*\label{code:catsdogs:4}
        maxActivation = np.argmax(predictions[0]) #*\label{code:catsdogs:5}
        predictedClassEntry = predictions[:, maxActivation] #*\label{code:catsdogs:6}

    grads = tape.gradient(predictedClassEntry, featureMaps) #*\label{code:catsdogs:8}
    pooledGrads = np.mean(grads, axis=(0, 1, 2)) #*\label{code:catsdogs:9}
    weightedFeatures = featureMaps * pooledGrads #*\label{code:catsdogs:10}
    heatmapImg = np.sum(weightedFeatures, axis=-1).squeeze() #*\label{code:catsdogs:17}

    heatmapImg = np.maximum(heatmapImg, 0) 
    if np.max(heatmapImg) > 0 : heatmapImg /= np.max(heatmapImg) 
    return heatmapImg, predictions.numpy()

from matplotlib import cm
from PIL import Image as PILImage
from tensorflow.keras.preprocessing import image

def overlayHeatmap(img, heatmapImg):
    heatmapImg = cm.Reds(heatmapImg)[..., :3] #*\label{code:catsdogs:11}
    heatmapImg = image.array_to_img(heatmapImg) #*\label{code:catsdogs:12}
    heatmapImg = heatmapImg.resize(img.shape[:-1], resample=PILImage.BICUBIC) #*\label{code:catsdogs:13}
    heatmapImg = image.img_to_array(heatmapImg) #*\label{code:catsdogs:14}
    
    imGray = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2] #*\label{code:catsdogs:15}
    imGray = cm.gray(imGray)[..., :3] #*\label{code:catsdogs:16}
    
    if np.max(imGray) <= 1: imGray = 255*imGray
    if np.max(heatmapImg) <= 1 : heatmapImg = 255*heatmapImg
    superimposedImg = np.minimum(heatmapImg * 0.6 + 0.2*imGray, 255).astype(np.uint8)
    return superimposedImg

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
imageList = ['hund1.jpg', 'hund2.jpg','hund3.jpg', 'hund4.jpg',
             'katze1.jpg', 'katze2.jpg', 'katze3.jpg', 'katze4.jpg', 
             'ratte.jpg', 'luchs.jpg', 'wolf.jpg', 'katze1freigestellt.jpg']
for imageFile in imageList:
    imgSize = CNN.input_shape[1:-1]
    img = image.load_img(imageFile, target_size=imgSize)
    img = image.img_to_array(img)/255.0
    plt.figure(); plt.imshow(img)
    hm, predictions = heatmap(img, CNN)
    plt.figure(); plt.imshow(hm, cmap=cm.Reds)
    if np.argmax(predictions[0]) == 0: classstring = 'cat'
    else: classstring = 'dog'
    print(imageFile,' predicted as ',classstring, ' with ', predictions)
    fusion = overlayHeatmap(img, hm)
    plt.figure(); plt.imshow(fusion)
    name = imageFile.split('.')[0]
    plt.title(name+str(predictions))
    name = 'heat'+name+'.png'
    image.array_to_img(fusion).save(name, 'PNG')