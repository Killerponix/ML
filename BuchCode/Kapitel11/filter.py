import numpy as np

def myConv(pic,strides=(1,1), convMatrix=np.array([ [1,0,-1],[0,0,0],[-1,0,1]])):
    size    = convMatrix.shape[0]
    pad     = size-1
    shift   = int((pad)/2)
    picPadding = np.zeros( (pic.shape[0]+pad, pic.shape[1]+pad) ) #*\label{code:filer:1}
    picPadding[shift:pic.shape[0]+shift,shift:pic.shape[1]+shift] = pic #*\label{code:filer:2}
    picConv = np.zeros_like(picPadding)
    for i in range(0,pic.shape[0],strides[0]):
        for j in range(0,pic.shape[1],strides[0]):
            picConv[i+shift,j+shift] = np.sum((convMatrix*picPadding[i:i+size,j:j+size])) #*\label{code:filer:0}
    picConv = picConv[shift:picConv.shape[0]-shift,shift:picConv.shape[1]-shift:] #*\label{code:filer:3}       
    picConv = picConv[0:picConv.shape[0]:strides[0],:]  #*\label{code:filer:4}      
    picConv = picConv[:,0:picConv.shape[1]:strides[1]]  #*\label{code:filer:5}
    return picConv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
pic = mpimg.imread('terryRatte.png')/255
convMatrix = np.ones( (9,9) )
convMatrix[4,4] = -convMatrix.size +1
picConv = myConv(pic,strides=(2,2), convMatrix=convMatrix)
plt.figure()
plt.imshow(pic, cmap='gray')
plt.figure()
plt.imshow(picConv, cmap='gray')

# skriptBeispiel = np.zeros((7,7))
# skriptBeispiel[1,1:6] = np.array([1, 0, 1, 2, 0])
# skriptBeispiel[2,1:6] = np.array([1, 0, 1, 2, 3])
# skriptBeispiel[3,1:6] = np.array([0, 0, 1, 2, 3])
# skriptBeispiel[4,1:6] = np.array([0, 0, 0, 1, 2])
# skriptBeispiel[5,1:6] = np.array([0, 0, 1, 2, 2])
# skriptCon = np.zeros_like(skriptBeispiel)

# convMatrix  = np.array([ [ 1,  0 , -1], [0, 0, 0], [-1, 0, 1]] )
# for i in range(0,5):
#     for j in range(0,5):
#         skriptCon[i+1,j+1] = np.sum((convMatrix*skriptBeispiel[i:i+3,j:j+3]))