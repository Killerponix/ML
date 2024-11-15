import numpy as np
def heavyside(x):
    y=np.ones_like(x,dtype=np.float)
    y[x <=0] = 0
    return y

def fit(X,Y):



    pass

def predict(x,w,xMin,xMax):
    xC =np.ones((x.shape[0],3))

    pass