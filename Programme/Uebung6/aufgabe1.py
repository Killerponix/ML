import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp



def sigmoid(x):
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid



fig, ax= plt.subplots()
# x= range(-10,10)
x= np.linspace(-30,30)
# x2=range(10,20)
# y=(4*sp.expit(0.05*x))-(1*sp.expit(25*x-25)) #sigmoid
# y2=sp.expit(x2)
ax.plot(x,y)
# ax.plot(x2,y2)
ax.grid()


# fig.show()
plt.show()

# fig2 = plt.figure()
# x=np.arange(-2,2,0.01)
# ax2 = fig2.add_subplot(2,2,1)
# y1=np.maximum(x,0)
# y2=np.maximum(-x,0)
# ax2.plot(x,y1)
# ax2.plot(x,y2)
# fig2.show()

