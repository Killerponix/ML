import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time 

plt.figure(1)
fuenf=mpimg.imread('356.png')
plt.imshow(fuenf, cmap='gray')
print(fuenf.shape)

plt.figure(2)
weka=mpimg.imread('wekaralle.png')
plt.imshow(weka)
print(weka.shape)

newWeka = np.copy(weka)
t = time.time()
for x in range(0,800): #*\label{code:basicsBilder1} 
    for y in range(0,800):
        newWeka[x,y,0] = max(1 - (x/400 - 1)**2 - (y/400-1)**2,0) #*\label{code:basicsBilder2}
elapsed = time.time() - t
print ("Benoetigte Zeit(s): " + str(elapsed))  
plt.figure(3)
plt.imshow(newWeka)

newWeka2 = np.copy(weka)
t = time.time()
xv, yv = np.meshgrid(np.arange(0, 800),np.arange(0, 800))
newWeka2[:,:,0] = np.maximum(1 - (xv/400 - 1)**2 - (yv/400-1)**2,0)
del(xv, yv)
elapsed = time.time() - t    
print ("Benoetigte Zeit(s): " + str(elapsed))
plt.figure(4)
plt.imshow(newWeka2[...,:3]@[0.299, 0.587, 0.114], cmap='gray')     
