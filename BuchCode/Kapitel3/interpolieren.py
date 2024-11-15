import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt 

A = np.zeros([22,2])
A[:,0] = np.arange(0, 43, 2)
A[0:11,1]     = [2, 6, 9, 12, 14, 16, 17.5, 18.5, 20, 20.5, 21.5]
A[11:22,1] = [22, 22.5, 22.7, 23.5, 23.5, 23.7, 24, 24, 24.2, 24.2, 24.5]
plt.plot(A[:,0], A[:,1], 'o', label="Messwerte", c='k')
plt.xlabel('Zeit [s]')
plt.ylabel('Spannung [V]')

p2 = interpolate.lagrange(A[[0 , 10 ,21 ],0], A[[0 , 10 ,21 ],1])
xnew = np.arange(-2, 50, 2)
ynew = p2(xnew)   
error = (( p2(A[:,0]) - A[:,1] )**2).sum()
print('P2 => Quadratische Fehler: %.4e; gemittelt %.4e.' % (error, error/22)) #*\label{lst:p:13:line:17}
plt.plot(xnew, ynew, label="Polynome Ordnung 2", linestyle='-', c='k')
p5 = interpolate.lagrange(A[0:22:4,0], A[0:22:4,1])
xnew = np.arange(-2, 50, 2)
ynew = p5(xnew)   
error = (( p5(A[:,0]) - A[:,1] )**2).sum()
print('P5 => Quaratische Fehler: %.4e; gemittelt %.4e.' % (error, error/22)) #*\label{lst:p:13:line:23}
plt.plot(xnew, ynew,label="Polynome Ordnung 5", linestyle='--', c='r') #*\label{buildMeshFilter4}

plt.legend(loc='lower right')
plt.show()