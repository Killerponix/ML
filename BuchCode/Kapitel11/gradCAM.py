import numpy as np
import tensorflow as tf    
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from matplotlib import cm
from PIL import Image as PILImage
from tensorflow.keras.preprocessing import image

def heatmap(img, model):
    for l in model.layers: 
        if isinstance(l, Conv2D): lastConvLayer = l 
    calcFeaturesAndPred = Model([model.input], [lastConvLayer.output, model.output])
    img = tf.Variable(img); #*\label{code:catsdogs:7}
    with tf.GradientTape() as tape:
        featureMaps, predictions = calcFeaturesAndPred(img[np.newaxis,...])  
        maxActivation = np.argmax(predictions[0]) 
        predictedClassEntry = predictions[:, maxActivation] 
    grads = tape.gradient(predictedClassEntry, featureMaps) 
    pooledGrads = np.mean(grads, axis=(0, 1, 2)) 
    weightedFeatures = featureMaps * pooledGrads 
    heatmapImg = np.sum(weightedFeatures, axis=-1).squeeze() 
    heatmapImg = np.maximum(heatmapImg, 0) 
    if np.max(heatmapImg) > 0 : heatmapImg /= np.max(heatmapImg) 
    return heatmapImg, predictions.numpy()



def overlayHeatmap(img, heatmapImg):
    heatmapImg = cm.Reds(heatmapImg)[..., :3] 
    heatmapImg = image.array_to_img(heatmapImg) 
    heatmapImg = heatmapImg.resize(img.shape[:-1], resample=PILImage.BICUBIC) 
    heatmapImg = image.img_to_array(heatmapImg) 
    
    imGray = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2] 
    imGray = cm.gray(imGray)[..., :3] 
    
    if np.max(imGray) <= 1: imGray = 255*imGray
    if np.max(heatmapImg) <= 1 : heatmapImg = 255*heatmapImg
    superimposedImg = np.minimum(heatmapImg * 0.6 + 0.2*imGray, 255).astype(np.uint8)
    return superimposedImg