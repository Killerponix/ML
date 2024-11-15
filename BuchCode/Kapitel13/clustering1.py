import numpy as np
import matplotlib.pyplot as plt

def bubbleSetNormal(mx,my,number,s):
    x = np.random.normal(0, s, number) + mx
    y = np.random.normal(0, s, number) + my
    return(x,y)

def fourBalls(n1,n2,n3,n4):
    np.random.seed(42)        
    dataset = np.zeros( (n1+n2+n3+n4,2) )    
    (dataset[0:n1,0],dataset[0:n1,1])         = bubbleSetNormal( 2.5, 1.0,n1,0.5)     
    (dataset[n1:n1+n2,0],dataset[n1:n1+n2,1]) = bubbleSetNormal( 2.0,-3.0,n2,0.3)    
    (dataset[n1+n2:n1+n2+n3,0],dataset[n1+n2:n1+n2+n3,1]) = bubbleSetNormal(-2.0, 5.0,n3,0.6)   
    (dataset[n1+n2+n3:n1+n2+n3+n4,0],dataset[n1+n2+n3:n1+n2+n3+n4,1]) = bubbleSetNormal(-4.0,-1.0,n4,0.9)        
    return (dataset)    

plt.close('all')
n1=n2=n3=n4=400
# Testbeispiel 1
XBalls = fourBalls(n1,n2,n3,n4)

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.scatter(XBalls[0:n1,0]  ,XBalls[0:n1,1],c='red',s=60,alpha=0.2,marker='*')
ax.scatter(XBalls[n1:n1+n2,0],XBalls[n1:n1+n2,1],c='blue',s=60,alpha=0.2,marker='s')
ax.scatter(XBalls[n1+n2:n1+n2+n3,0],XBalls[n1+n2:n1+n2+n3,1],c='black',s=60,alpha=0.2,marker='<')
ax.scatter(XBalls[n1+n2+n3:n1+n2+n3+n4,0],XBalls[n1+n2+n3:n1+n2+n3+n4,1],c='green',s=60,alpha=0.2,marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True,linestyle='-',color='0.75')
ax.set_aspect('equal', 'datalim')
ax.set_title("Vier gleich grosse Mengen")

n1=n2=100
n3=n4=400
# Testbeispiel 2
XBalls2 = fourBalls(n1,n2,n3,n4)

fig = plt.figure(2)
ax = fig.add_subplot(1,1,1)
ax.scatter(XBalls2[0:n1,0]  ,XBalls2[0:n1,1],c='red',s=60,alpha=0.2,marker='*')
ax.scatter(XBalls2[n1:n1+n2,0],XBalls2[n1:n1+n2,1],c='blue',s=60,alpha=0.2,marker='s')
ax.scatter(XBalls2[n1+n2:n1+n2+n3,0],XBalls2[n1+n2:n1+n2+n3,1],c='black',s=60,alpha=0.2,marker='<')
ax.scatter(XBalls2[n1+n2+n3:n1+n2+n3+n4,0],XBalls2[n1+n2+n3:n1+n2+n3+n4,1],c='green',s=60,alpha=0.2,marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True,linestyle='-',color='0.75')
ax.set_aspect('equal', 'datalim')
ax.set_title("Vier unterschiedlich grosse Mengen")

def mouseShape():
    np.random.seed(42)        
    dataset = np.zeros( (1000,2) )    
    (dataset[0:150,0],dataset[0:150,1])       = bubbleSetNormal(-0.75, 0.75,150,0.15)     
    (dataset[150:300,0],dataset[150:300,1])   = bubbleSetNormal( 0.75, 0.75,150,0.15)    
    (dataset[300:1000,0],dataset[300:1000,1]) = bubbleSetNormal( 0, 0,700,0.29)   
    return (dataset) 
# Testbeispiel 3
XBMouse = mouseShape()

fig = plt.figure(3)
ax = fig.add_subplot(1,1,1)
ax.scatter(XBMouse[0:150,0]  ,XBMouse[0:150,1],c='red',s=60,alpha=0.2,marker='*')
ax.scatter(XBMouse[150:300,0],XBMouse[150:300,1],c='blue',s=60,alpha=0.2,marker='s')
ax.scatter(XBMouse[300:1000,0],XBMouse[300:1000,1],c='black',s=60,alpha=0.2,marker='<')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True,linestyle='-',color='0.75')
ax.set_aspect('equal', 'datalim')
ax.set_title("Mouse Set")

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
# Testbeispiel 4
(XMoons,_) = twoMoonsProblem()

fig = plt.figure(4)
ax = fig.add_subplot(1,1,1)
ax.scatter(XMoons[0:240,0]  ,XMoons[0:240,1],c='red',s=60,alpha=0.2,marker='*')
ax.scatter(XMoons[240:480,0],XMoons[240:480,1],c='black',s=60,alpha=0.2,marker='<')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True,linestyle='-',color='0.75')
ax.set_aspect('equal', 'datalim')
ax.set_title("Two Moons Set")

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

fig = plt.figure(5)
ax = fig.add_subplot(1,1,1)
ax.scatter(Xcircels[0:800,0]   ,Xcircels[0:800,1],c='red',s=60,alpha=0.2,marker='*')
ax.scatter(Xcircels[800:1600,0],Xcircels[800:1600,1],c='black',s=60,alpha=0.2,marker='<')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True,linestyle='-',color='0.75')
ax.set_aspect('equal', 'datalim')
ax.set_title("Zwei Kreise")
    
np.random.seed(42) 
# Testbeispiel 6    
XRauschen = np.random.random( (1000,2) )
fig = plt.figure(6)
ax = fig.add_subplot(1,1,1)
ax.scatter(XRauschen[0:1000,0]   ,XRauschen[0:1000,1],c='red',s=60,alpha=0.2,marker='*')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True,linestyle='-',color='0.75')
ax.set_aspect('equal', 'datalim')
ax.set_title("Rauschen")


def kmeans(X, noOfClusters,maxIterations=42,p=2):
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    newcentres = np.random.rand(noOfClusters,X.shape[1])*(Xmax - Xmin)+Xmin
    oldCentres = np.random.rand(noOfClusters,X.shape[1])*(Xmax - Xmin)+Xmin
    count = 0
    
    while np.sum(np.sum(oldCentres-newcentres))!= 0 and count<maxIterations:
        count = count + 1
        oldCentres = np.copy(newcentres)
        distances = ( np.sum( np.abs(X-newcentres[0,:])**p,axis=1) )**(1/p)
        for j in range(1,noOfClusters):
            distancesAdd = ( np.sum( np.abs(X-newcentres[j,:])**p,axis=1) )**(1/p)
            distances = np.vstack( (distances,distancesAdd) )
        cluster = distances.argmin(axis=0)

        for j in range(noOfClusters):
            index = np.flatnonzero(cluster == j)    
            if index.shape[0]>0:
                newcentres[j,:] = np.sum(X[index,:],axis=0)/index.shape[0]
            else:
                distances = ( np.sum( (X-newcentres[j,:])**p,axis=1) )**(1/p)
                i = distances.argmin(axis=0)
                newcentres[j,:] = X[i,:] 
                cluster[i]=j
    return(newcentres,cluster)

np.random.seed(13456) 
(newcentres,cluster) = kmeans(XBalls2,4,p=2)
print(newcentres)
fig = plt.figure(7)
ax = fig.add_subplot(1,1,1)
markers=['*', 's', '<', 'o', 'X', '8', 'p', 'h', 'H', 'D', 'd', 'P']
colorName = ['red','blue','black','green']
for j in range(4):
    index = np.flatnonzero(cluster == j)
    ax.scatter(XBalls2[index,0]  ,XBalls2[index,1],c=colorName[j],s=60,alpha=0.2,marker=markers[j])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True,linestyle='-',color='0.75')
ax.set_aspect('equal', 'datalim')
