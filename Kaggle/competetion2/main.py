import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

xtrain = np.load("XTrain.npy")
xTest = np.load("XTest.npy")
yTrain = np.load("YTrain.npy")

myDevice = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices= myDevice, device_type='GPU')

print(xtrain)