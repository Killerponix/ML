import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

reducePicDim = 256

trainDatagen = ImageDataGenerator(rotation_range=30, rescale=1./255, horizontal_flip=0.1)
trainGenerator = trainDatagen.flow_from_directory(
    directory=r"./dogs-vs-cats/train/", target_size=(reducePicDim, reducePicDim),
    color_mode="rgb", batch_size=16, class_mode="categorical", shuffle=True, seed=42)

try:
    CNN = load_model("dogVScatVGGjustDense.h5")
except:
    stumpVGG16 = VGG16(weights='imagenet', include_top=False,
                  input_shape=(reducePicDim, reducePicDim, 3))
    stumpVGG16.trainable=False  
    flat = Flatten()(stumpVGG16.output) #*\label{code:vggreuse:1}
    x = Dense(100, activation='relu', name="dense1CatsVsDogs")(flat)
    x = Dense(50, activation='relu', name="dense2CatsVsDogs")(x)
    output = Dense(2, activation='softmax', name="softmaxCatsVsDogs")(x)
    CNN = Model(stumpVGG16.inputs, output, name='VGGCatDog')  #*\label{code:vggreuse:2}
            
    CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    CNN.fit_generator(generator=trainGenerator, epochs=5, verbose=True)
    CNN.save("dogVScatVGGjustDense.h5")

testDatagen = ImageDataGenerator(rescale=1./255)
testGenerator = testDatagen.flow_from_directory(
    directory=r"./dogs-vs-cats/test/", target_size=(reducePicDim, reducePicDim),
    color_mode="rgb", batch_size=8, class_mode="categorical", shuffle=False)
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

from gradCAM import heatmap, overlayHeatmap
from matplotlib import cm
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
    name = 'heat'+name+'vgg16dense.png'
    image.array_to_img(fusion).save(name, 'PNG')

try:
    CNN = load_model("dogVScatVGGPlus.h5")
except:
    from tensorflow.keras.optimizers import Adam
    CNN = load_model("dogVScatVGGjustDense.h5")  
    slowADAM = Adam(learning_rate=0.0001)
    for i in range(0,23): CNN.layers[i].trainable=True #*\label{code:vggreuse:3}
    CNN.compile(optimizer=slowADAM,loss='categorical_crossentropy',metrics=['accuracy'])
    CNN.summary()
    CNN.fit_generator(generator=trainGenerator, epochs=5, verbose=True)
    CNN.save("dogVScatVGGPlus.h5")
    
testGenerator.reset()
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
    name = 'heat'+name+'vgg16allTrain.png'
    image.array_to_img(fusion).save(name, 'PNG')