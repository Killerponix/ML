import numpy as np

np.random.seed(42)

dataset = np.loadtxt("Autoklassifizierung.csv", delimiter=",")


y = dataset[:,0]
x = np.ones( (len(y),3) ) #*\label{code:heblernen1}
x[:,0:2] = dataset[:,1:3] #*\label{code:heblernen2}

xMin = x[:,0:2].min(axis=0); xMax = x[:,0:2].max(axis=0) 
x[:,0:2] = (x[:,0:2] - xMin) / (xMax - xMin) #*\label{code:heblernen3}
t = 0; tmax=100000 
eta = 0.25 
Dw = np.zeros(3)
w = np.random.rand(3) - 0.5
convergenz = 1

def myHeaviside(x):
    y = np.ones_like(x,dtype=np.float)
    y[x <= 0] = 0 
    return(y)
    
while (convergenz > 0) and (t<tmax): 
    t = t +1;
    WaehleBeispiel = np.random.randint(len(y))
    xB = x[WaehleBeispiel,:].T
    yB = y[WaehleBeispiel]
    error = yB - myHeaviside(w@xB)
    for j in range(len(xB)):
        Dw[j]= eta*error*xB[j]
        w[j] = w[j] + Dw[j]
    convergenz =  np.linalg.norm(y-myHeaviside(w@x.T))

def predict(x,w,xMin,xMax):
    xC = np.ones( (x.shape[0],3) )
    xC[:,0:2] = x
    xC[:,0:2] = (xC[:,0:2] - xMin) / (xMax - xMin); print(xC)
    y = w@xC.T
    y[y>0] = 1
    y[y<= 0] = 0
    return(y)
# SEAT	Ibiza, Sokda Octavia, Toyota Avensis und Yaris GRMN
xTest = np.array([[12490, 48], [31590, 169],[24740, 97], [30800, 156]])
yPredict = predict(xTest,w,xMin,xMax)
print(yPredict)

import matplotlib.pyplot as plt
from matplotlib import cm
a= np.linspace(-1, 1, 50)
b=-w[0]/w[1]*a-w[2]/w[1]
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.plot(a,b,'k', linewidth=1.5, linestyle='dashed')
indexA = np.flatnonzero(y>0.5)
indexB = np.flatnonzero(y<0.5)
ax.scatter(x[indexA,0],x[indexA,1],color='red', marker='o')
ax.scatter(x[indexB,0],x[indexB,1],color='black', marker='+')
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_ylim([-0.25,1.25])
ax.set_xlim([0,1])
ax.set_title("Berechnet mit Random Seed 42")

xBool = np.array([[1, 0],[0, 1],[1, 1],[0, 0]])
w = np.array([1, 1, -0.5])
print(predict(xBool,w,0,1))
w = np.array([1, 1, -1.5])
print(predict(xBool,w,0,1))