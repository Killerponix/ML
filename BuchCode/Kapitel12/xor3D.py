import numpy as np
np.random.seed(42)

xO = np.array([ [0,0,1], [0,1,0],[1,0,0],[1,1,0]])
y = np.array([0, 1, 1, 0])
x = np.ones( (len(y),4) )
x[:,0:3] = xO

def myHeaviside(x):
    y = np.ones_like(x,dtype=np.float)
    y[x <= 0] = 0 
    return(y)

t = 0; tmax=100000 
eta = 0.25 
Dw = np.zeros(4)
w = np.random.rand(4) - 0.5
convergenz = 1    
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

def predict(x,w):
    xC = np.ones( (x.shape[0],4) )
    xC[:,0:3] = x
    y = w@xC.T
    y[y>0] = 1
    y[y<= 0] = 0
    return(y)

yPredict = predict(xO,w)
print(yPredict)
