import numpy as np 

def softmax(z):
    return(np.exp(z)/np.sum(np.exp(z))) 

def crossEntropy(p,q):
    return( -np.sum(p*np.log(q)))

mseZ = 0.0
mseP = 0.0
croE = 0.0

r = np.zeros((3,3))
r[0,0] = 1; r[1,1]= 1; r[2,0] = 1
z = np.zeros((3,3))

z[0] = np.array([0.9, 0.2, 0.1])
z[1] = np.array([0.1, 0.9, 0.2])
z[2] = np.array([0.7, 0.3, 0.5])

for i in range(3):
    mseZ += np.sum( (r[i]-z[i])**2)
    p = softmax(z[i])
    mseP += np.sum( (r[i]-p)**2)
    croE += crossEntropy(r[i],z[i])
    print(p.round(2)  , np.sum(p))    

mseZ = mseZ/9
mseP = mseP/9
croE = croE/9
print(mseZ.round(2),mseP.round(2),croE.round(2))


z[0] = np.array([0.6, 0.1, 0.1])
z[1] = np.array([0.2, 0.8, 0.1])
z[2] = np.array([0.5, 0.4, 0.4])

for i in range(3):
    mseZ += np.sum( (r[i]-z[i])**2)
    p = softmax(z[i])
    mseP += np.sum( (r[i]-p)**2)
    croE += crossEntropy(r[i],z[i])
    print(p.round(2)  , np.sum(p))     

mseZ = mseZ/9
mseP = mseP/9
croE = croE/9
print(mseZ.round(2),mseP.round(2),croE.round(2))
