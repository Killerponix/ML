import numpy as np

x = np.arange(0,1.1,0.1)
y1 = 0.5*x
y2 = (x-0.5)**2 
y3 = np.log(x+1)

data = np.vstack( (x,y1,y2,y3) )
kor = np.corrcoef(data)
print(kor[0,:])

y4 = x**2 
data = np.vstack( (x,y4) )
kor = np.corrcoef(data)
print(kor[0,1])