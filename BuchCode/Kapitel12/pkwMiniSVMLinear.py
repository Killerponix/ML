import numpy as np
from sklearn import svm

fFloat  = open("Autoklassifizierung.csv","r")
dataset = np.loadtxt(fFloat, delimiter=",")
fFloat.close()
y = dataset[:,0]
x = dataset[:,1:3] 
xMin = x.min(axis=0); xMax = x.max(axis=0) 
x = (x - xMin) / (xMax - xMin)

svmLin = svm.SVC(kernel='linear', C=100) #*\label{code:svm:linear:0}
svmLin.fit(x,y)

w = svmLin.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-1, 1,50)
yy = a * xx - (svmLin.intercept_[0]) / w[1]
margin = 1 / np.sqrt(np.sum(svmLin.coef_ ** 2))  #*\label{code:svm:linear:1}
yMarginDown = yy - np.sqrt(1 + a ** 2) * margin
yMarginUp   = yy + np.sqrt(1 + a ** 2) * margin

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xx,yy,'k', linewidth=1.5)
plt.plot(xx, yMarginDown, 'k--')
plt.plot(xx, yMarginUp, 'k--')
iA = np.flatnonzero(y==1)
iB = np.flatnonzero(y==0)
ax.scatter(x[iA,0],x[iA,1],alpha=0.6,c='red', marker='+', linewidth=1.5)
ax.scatter(x[iB,0],x[iB,1],alpha=0.6,c='black', marker='o', linewidth=1.5)
ax.set_xlabel('$x_0$'); ax.set_ylabel('$x_1$')
ax.set_ylim([-0.25,1.25]); ax.set_xlim([0,1])
ax.set_title("Berechnet mit C=100")