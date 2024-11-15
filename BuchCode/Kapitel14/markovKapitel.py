import numpy as np

delta = np.zeros( (9,9) )
delta[3,6] = delta[3,0] = delta[2,1] = delta[1,0] = delta[0,1] = delta[0,3] = 0.5
delta[6,3] = delta[5,8] = delta[7,6] = delta[7,8] = 0.5
delta[4,4] = delta[2,2] = 1
delta[8,7] = delta[8,5] = delta[6,7] = delta[4,5] = delta[4,7] = delta[2,5] = 1/3

s = np.zeros( (9,1) )
s[0] = 1 
sNew = np.linalg.matrix_power(delta,5)@s

sNew = np.linalg.matrix_power(delta,100)@s
print(sNew)

s = np.zeros( (9,1) )
s[6] = 1 
sNew = np.linalg.matrix_power(delta,100)@s
print(sNew)