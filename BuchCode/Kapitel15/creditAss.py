import numpy as np

gamma = 0.9
n = 6
r = np.zeros(n,dtype=float); r[-1] = 1
Q = np.zeros_like(r,dtype=float)
for i in range(10):
    QLast = Q.copy()
    Q[0:len(r)-1] = r[0:len(r)-1] + gamma*Q[1:len(r)]
    Q[len(r)-1]   = r[len(r)-1]   + gamma*Q[0] #*\label{code:dorornotdone:0}
    diff = np.mean(np.abs(Q - QLast))
    print(i, diff, Q)

r = np.zeros(n,dtype=float); r[-1] = 1
Q = np.zeros_like(r,dtype=float)
for i in range(10):
    QLast = Q.copy()
    Q[0:len(r)-1] = r[0:len(r)-1] + gamma*Q[1:len(r)]
    Q[len(r)-1]   = r[len(r)-1]   #*\label{code:dorornotdone:1}
    diff = np.mean(np.abs(Q - QLast))
    print(i, diff, Q)

from numpy.linalg import matrix_power
n = 100
steps= 10000
D = 0.5*np.eye(n,k=1) + 0.5*np.eye(n,k=-1)
D[ 1, 0] = 1; D[-1,-1] = 1; D[-2,-1] = 0
s = np.zeros(n); s[1] = 1
stepList = [100,500,1000,5000,10000,20000,30000]
p = []
for steps in stepList:
    finals = matrix_power(D,steps)@s
    p.append(finals[-1])
import matplotlib.pyplot as plt
plt.plot(stepList,p,c='k')
plt.xlabel('Scrhitte')
plt.ylabel('Wahrscheinlichkeit für das Erreichen von $S_n$')
plt.figure()
n = 100
m = 50
D = np.zeros( (n+m,n+m))
D[0:n,0:n] = 0.5*np.eye(n,k=1) + 0.5*np.eye(n,k=-1)
D[n:n+m,n:n+m] = 0.05*np.eye(m,k=1) + 0.95*np.eye(m,k=-1)
D[0,1] = 1.0/30; D[2,1] = 1.0/30
D[n+1,1] = 0.9+ 1.0/30
D[n,n] = 1; D[1, 0] = 1
D[n+1,n] = 0; D[1,n+1] = 0.05
D[1,-1] = 1; D[-2,-1] = 0
D[n,n-1] = 0.5; D[n,n+1] = 0 
s = np.zeros(n+m); s[1] = 1
stepList = [500,1000,5000,10000,20000,30000,50000,75000,100000,150000,200000]
p = []
for steps in stepList:
    finals = matrix_power(D,steps)@s
    p.append(finals[n])
import matplotlib.pyplot as plt
plt.plot(stepList,p,c='k')
plt.xlabel('Scrhitte')
plt.ylabel('Wahrscheinlichkeit für das Erreichen von $S_n$')


# np.random.seed(42)
# m = 25
# n = 100 
# runs = 100
# steps = np.zeros(runs) 
# for i in range(runs):
#     S = 2
#     while S != n:
#         steps[i] = steps[i]+1
#         if S == 1:
#             if np.random.rand() < 0.5 : 
#                 S=2
#             else:
#                 S=1
#                 #steps[i] = steps[i] + m #*\label{code:dorornotdone:3}
#         else:
#             coin = np.random.rand()
#             if coin < 0.5 : S = S -1
#             else: S = S + 1
# print(np.median(steps), np.mean(steps), np.std(steps))