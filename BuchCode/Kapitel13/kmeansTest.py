import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans

def bubbleSetNormal(mx,my,number,s):
    x = np.random.normal(0, s, number) + mx
    y = np.random.normal(0, s, number) + my
    return(x,y)

def fourBalls(n1,n2,n3,n4):
    np.random.seed(42)        
    dataset = np.zeros( (n1+n2+n3+n4,2) )    
    (dataset[0:n1,0],dataset[0:n1,1])     = bubbleSetNormal( 2.5, 1.0,n1,0.5)     
    (dataset[n1:n1+n2,0],dataset[n1:n1+n2,1]) = bubbleSetNormal( 2.0,-3.0,n2,0.3)    
    (dataset[n1+n2:n1+n2+n3,0],dataset[n1+n2:n1+n2+n3,1]) = bubbleSetNormal(-2.0, 5.0,n3,0.6)   
    (dataset[n1+n2+n3:n1+n2+n3+n4,0],dataset[n1+n2+n3:n1+n2+n3+n4,1]) = bubbleSetNormal(-4.0,-1.0,n4,0.9)        
    return (dataset)    

np.random.seed(42) 
markers=['*', 's', '<', 'o', 'X', '8', 'p', 'h', 'H', 'D', 'd', 'P']
colorName = ['red','black','blue','green']   

plt.close('all')
n1=n2=n3=n4=400
# Testbeispiel 1
XBalls = fourBalls(n1,n2,n3,n4)

clusterAlg= kmeans(4)

(_, c) = clusterAlg.fit(XBalls)

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)

for i in range(0,c.max()+1):
    index = np.flatnonzero(c == i)
    ax.scatter(XBalls[index,0],XBalls[index,1],c=colorName[i],s=60,alpha=0.2,marker=markers[i])
ax.set_title('noOfClusters = 4')

# -------------------------------------------------------------- 
n1=n2=100
n3=n4=400
# Testbeispiel 2
XBalls2 = fourBalls(n1,n2,n3,n4)

clusterAlg = kmeans(4)

(_, c) = clusterAlg.fit(XBalls2)

fig = plt.figure(4)
ax = fig.add_subplot(1,1,1)

for i in range(0,c.max()+1):
    index = np.flatnonzero(c == i)
    ax.scatter(XBalls2[index,0],XBalls2[index,1],c=colorName[i],s=60,alpha=0.2,marker=markers[i])
ax.set_title('noOfClusters = 4')

# -------------------------------------------------------------- 

def mouseShape():
    np.random.seed(42)        
    dataset = np.zeros( (1000,2) )    
    (dataset[0:150,0],dataset[0:150,1])     = bubbleSetNormal(-0.75, 0.75,150,0.15)     
    (dataset[150:300,0],dataset[150:300,1]) = bubbleSetNormal( 0.75, 0.75,150,0.15)    
    (dataset[300:1000,0],dataset[300:1000,1]) = bubbleSetNormal( 0, 0,700,0.29)   
    return (dataset) 
# Testbeispiel 3
XBMouse = mouseShape()

clusterAlg = kmeans(3)
(_, c) = clusterAlg.fit(XBMouse)

fig = plt.figure(6)
ax = fig.add_subplot(1,1,1)

for i in range(0,c.max()+1):
    index = np.flatnonzero(c == i)
    ax.scatter(XBMouse[index,0],XBMouse[index,1],c=colorName[i],s=60,alpha=0.2,marker=markers[i])
ax.set_title('noOfClusters = 3')

# -------------------------------------------------------------- 
def twoMoonsProblem( SamplesPerMoon=240, pNoise=2):
    np.random.seed(42) 
    tMoon0 = np.linspace(0, np.pi, SamplesPerMoon)
    tMoon1 = np.linspace(0, np.pi, SamplesPerMoon)
    Moon0x = np.cos(tMoon0)
    Moon0y = np.sin(tMoon0)
    Moon1x = 1 - np.cos(tMoon1)
    Moon1y = 0.5 - np.sin(tMoon1) 
    X = np.vstack((np.append(Moon0x, Moon1x), np.append(Moon0y, Moon1y))).T
    X = X + pNoise/100*np.random.normal(size=X.shape)
    Y = np.hstack([np.zeros(SamplesPerMoon), np.ones(SamplesPerMoon)])
    return X, Y

(XMoons,_) = twoMoonsProblem()

clusterAlg = kmeans(2)

(_, c) = clusterAlg.fit(XMoons)

fig = plt.figure(8)
ax = fig.add_subplot(1,1,1)

for i in range(0,c.max()+1):
    index = np.flatnonzero(c == i)
    ax.scatter(XMoons[index,0],XMoons[index,1],c=colorName[i],s=60,alpha=0.2,marker=markers[i])
ax.set_title('noOfClusters = 2')
# -------------------------------------------------------------- 

def circels():
    np.random.seed(42) 
    phi = np.linspace(0,2*np.pi, 800)
    x1 = 1.5*np.cos(phi)
    y1 = 1.5*np.sin(phi) 
    x2 = 0.5*np.cos(phi)
    y2 = 0.5*np.sin(phi)
    X = np.vstack((np.append(x1,x2), np.append(y1,y2))).T
    X = X + 0.1*np.random.normal(size=X.shape)
    return(X)
# Testbeispiel 5    
Xcircels = circels()

clusterAlg = kmeans(2)
(_, c) = clusterAlg.fit(Xcircels)

fig = plt.figure(10)
ax = fig.add_subplot(1,1,1)

for i in range(0,c.max()+1):
    index = np.flatnonzero(c == i)
    ax.scatter(Xcircels[index,0],Xcircels[index,1],c=colorName[i],s=60,alpha=0.2,marker=markers[i])
ax.set_title('noOfClusters = 2')
# -------------------------------------------------------------- 
np.random.seed(42) 
# Testbeispiel 6    
XRauschen = np.random.random( (1000,2) )

clusterAlg = kmeans(2) 
(_, c) = clusterAlg.fit(XRauschen)

fig = plt.figure(12)
ax = fig.add_subplot(1,1,1)

for i in range(0,c.max()+1):
    index = np.flatnonzero(c == i)
    ax.scatter(XRauschen[index,0],XRauschen[index,1],c=colorName[i],s=60,alpha=0.2,marker=markers[i])
ax.set_title('noOfClusters = 2')