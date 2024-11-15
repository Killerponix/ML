import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)
x = np.random.rand(50)
x = np.hstack( (x,x) )
y = 2*x - 0.5 
noise = 0.2*np.random.normal(size=x.shape[0])
ym = y + noise 
plt.plot(x,y,color='k')
r = np.array( [ [x[95],x[95]],[ym[95],y[95]] ] )
plt.plot(r[0,:],r[1,:], 'k:' )
plt.scatter(x,ym,color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()