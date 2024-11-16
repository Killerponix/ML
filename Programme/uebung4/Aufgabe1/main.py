import numpy as np
import sys
sys.path.append("C:/Users/Simon/Desktop/Studium/5.Semester/Maschinelles Lernen/ML/Programme/Model")
from Programme.Model.util.TrainTestSplit import *

def heavyside(x):
    y=np.ones_like(x,dtype=np.float)
    y[x <=0] = 0
    return y

def fit(X,Y):
    t=0; tmax=10000
    Dw= np.zeros(3)
    convergenz=1
    eta =0.25
    w=np.random.rand(3)-0.5
    while(convergenz>0) and (t<tmax):
        t=t+1
        rnd = np.random.randint(len(Y))
        xB =X[rnd,:].T
        yB= [rnd]
        error = yB -heavyside(w@xB)
        for j in range(len(xB)):
            Dw[j] = eta*error*xB[j]
            w[j]=w[j]+Dw[j]
            convergenz = np.linalg.norm(Y-heavyside(w@X.T))

def predict(x,w,xMin,xMax):
    xC =np.ones((x.shape[0],3))
    xC[:,0:2]=x
    xC[:,0:2]= (xC[:,0:2]-xMin)  / (xMax-xMin); print(xC)
    y= w@xC.T
    y[y>0]=1
    y[y<=0]=0
    return y

if __name__ == '__main__':
    data = np.loadtxt(delimiter=",",fname="Autoklassifizierung.csv")
    X = data[:,:1]
    Y = data[:,1:]
    x_train, x_test,y_train,y_test = train_test_split(X,Y)
    fit(x_train,y_train)
    predict(x_test)

