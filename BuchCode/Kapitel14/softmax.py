import numpy as np
import random
  
def softmax(Q, tau):
    return np.exp(Q/tau) / sum(np.exp(Q/tau))

Q=np.array([-10, 2.1, 1.9])
print( softmax(Q,0.1) )
print( softmax(Q,1.0) )
print( softmax(Q,100) )

def chooseAction(Q, tau=1):
    p = softmax(Q,1.0)
    toChoose = np.arange(0,len(Q))
    a = int(random.choices(toChoose,weights=p,k=1)[0])
    return(a)
    
random.seed(42)
counter = np.zeros(len(Q))
for i in range(10000):
    counter[chooseAction(Q)] += 1
print(counter)