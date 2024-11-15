import matplotlib.pyplot as plt
import csv
import numpy as np


def readTrafficSigns(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

trafficSignsImages, trafficSignsLabels = readTrafficSigns('Schildererkennung/Schildererkennung/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')

f = open('Schildererkennung/Schildererkennung/GTSRB_Final_Test_GT/GT-final_test.csv') 
header = f.readline().rstrip('\n')  
featureNames = header.split(';')
dataset = np.loadtxt(f, usecols=(1,2,3,4,5,6,7), delimiter=";")
f.close()

yTest = np.array(dataset[:,6],dtype=int)

def readTrafficSignsTest(prefix):
    images = [] 
    for row in range(len(yTest)):
        bildName = prefix +"/" + str(row).zfill(5) + ".ppm"
        images.append(plt.imread(bildName)) 
    return images

trafficSignsImagesTest = readTrafficSignsTest('Schildererkennung/Schildererkennung/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')

plt.imshow(trafficSignsImages[4000])
plt.show()

maxX = 48
maxY = 48



