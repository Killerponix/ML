import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def maxtrix2cartesian(row,col, maxrow):
    #  0123456
    # 0
    # 1
    # 2
    
    x = col.T
    y = (maxrow - row).T
    return x, y 

def cartesian2maxtrix(x,y, maxrow):
    # 2
    # 1
    # 0
    #  0123456
    
    col = x.T
    row = (maxrow - y).T 

    return row, col 


def plotQFunction(agent, env,epochen, saveFig = True, colorMax = 1, colorMin = -1.5):
    xGrid = len(env.world[0])
    yGrid = len(env.world)
    maxrow = yGrid - 1
    myShape = (yGrid,xGrid)
    
    plotX, plotY = np.mgrid[0:xGrid:1, 0:yGrid:1]
    rows, cols = cartesian2maxtrix(plotX, plotY, maxrow)
    Z = np.zeros( myShape, dtype=float)
    pW = np.zeros( (4,myShape[0],myShape[1]), dtype=float)
    A = np.zeros_like(Z, dtype=float)
    QMax = np.ones_like(Z, dtype=float)*-100000
    M = np.zeros_like(Z, dtype=float)
    r = rows.ravel()
    c = cols.ravel()
    Q = []
    for a in range(4):
        Q.append(Z.ravel().copy())
    del Z 
    
    # plotting the q-Values for all four directions 
    fig = plt.figure()
    fig.suptitle(str(epochen)+' Epochen')        
    actions = ['South', 'East', 'North', 'West']
    for a in range(4):
        for i in range(Q[a].shape[0]):
            obs = np.array((c[i],r[i]))
            Q[a][i] = agent.QFunction.predict(obs,a)
        ZZ = Q[a].reshape(myShape[0],myShape[1])
        A[ZZ>QMax] = a # uses in second plot
        QMax  = np.maximum(QMax,ZZ)
        ax = fig.add_subplot(2,2,a+1);   
        im = ax.pcolormesh(plotX, plotY,  ZZ.T, cmap=plt.cm.hot, vmin=colorMin, vmax=colorMax)
        ax.set_xlim(1,xGrid-1)
        ax.set_ylim(1,yGrid-1)
        ax.set_aspect('equal')
        ax.set_title(actions[a])
        cbar = plt.colorbar(im)
        cbar.set_label('Q-Value')
    if saveFig: 
        plt.savefig('vales'+str(epochen)+'.pdf')
        plt.savefig('vales'+str(epochen)+'.png')
        
   # setting values for the maze fields     
    M = M.ravel()
    for i in range(M.shape[0]):
        if env.world[r[i]][c[i]] == 'W' : 
            M[i] = 0.75
        elif env.world[r[i]][c[i]] == '-':
            M[i] = 0.2
        elif env.world[r[i]][c[i]] == '+':
            M[i] = 0.0
        elif env.world[r[i]][c[i]] == 'g':
            M[i] = 1        
    M = M.reshape(myShape[0],myShape[1])

    U = np.zeros( myShape, dtype=float)
    V = np.zeros( myShape, dtype=float)
    # pure arrow plot based on max value 
    fig = plt.figure()
    U[A==0] =  0 # south
    V[A==0] = -0.5
    U[A==2] =  0 # north 
    V[A==2] = +0.5
    U[A==1] = +0.5 # east
    V[A==1] =  0
    U[A==3] = -0.5 # west 
    V[A==3] =  0
    ax = fig.add_subplot(1,1,1)
    im = ax.pcolormesh(plotX, plotY, M.T, cmap=plt.cm.Greys)
    ax.quiver(cols+0.5, maxrow-rows+0.5, U, V, units='width')
    ax.set_xticks(np.arange(1,xGrid)) 
    ax.set_yticks(np.arange(1,yGrid))
    ax.set_xlim(1,xGrid-1)
    ax.set_ylim(1,yGrid-1)
    ax.set_title(str(epochen)+' Epochen')
    ax.set_aspect('equal')
    plt.show()
    
    if saveFig: plt.savefig('Directions'+str(epochen)+'.pdf')  

    if hasattr(agent, 'tau'):
        tau = agent.tau
    else:
        tau = 0.5
        
    # ploting the wind rose of propablities 
    for a in range(4):
        Q[a] = Q[a].reshape(myShape[0],myShape[1])
    for i in range(myShape[0]):
        for j in range(myShape[1]):
            qvalues = np.array([Q[0][i,j], Q[1][i,j], Q[2][i,j], Q[3][i,j] ] )
            pw = np.exp(qvalues/tau) / np.sum(np.exp(qvalues/tau))
            for a in range(4):
                pW[a,i,j] = pw[a] 
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.pcolormesh(plotX, plotY, M.T, cmap=plt.cm.Greys)

    U = np.zeros( myShape, dtype=float)
    V = np.zeros( myShape, dtype=float)
    for i in range(r.shape[0]):
        U[r[i],c[i]] =  0 # south
        V[r[i],c[i]] = -0.5*pW[0,r[i],c[i]]
    ax.quiver(cols+0.5, maxrow-rows+0.5, U, V,angles='xy', scale_units='xy', scale=1, headwidth = 2, color='g')
    U = np.zeros( myShape, dtype=float)
    V = np.zeros( myShape, dtype=float)
    for i in range(r.shape[0]):
        U[r[i],c[i]] =  0 # north 
        V[r[i],c[i]] = +0.5*pW[2,r[i],c[i]]
    ax.quiver(cols+0.5, maxrow-rows+0.5, U, V,angles='xy', scale_units='xy', scale=1, headwidth = 2, color='b')
    U = np.zeros( myShape, dtype=float)
    V = np.zeros( myShape, dtype=float)
    for i in range(r.shape[0]):
        U[r[i],c[i]] = +0.5*pW[1,r[i],c[i]] # east
        V[r[i],c[i]] =  0
    ax.quiver(cols+0.5, maxrow-rows+0.5, U, V,angles='xy', scale_units='xy', scale=1, headwidth = 2, color='g')        
    U = np.zeros( myShape, dtype=float)
    V = np.zeros( myShape, dtype=float)
    for i in range(r.shape[0]):
        U[r[i],c[i]] = -0.5*pW[3,r[i],c[i]] # west 
        V[r[i],c[i]] =  0
    ax.quiver(cols+0.5, maxrow-rows+0.5, U, V,angles='xy', scale_units='xy', scale=1, headwidth = 2, color='r')        

    ax.set_xticks(np.arange(1,xGrid)) 
    ax.set_yticks(np.arange(1,yGrid))
    ax.set_xlim(1,xGrid-1)
    ax.set_ylim(1,yGrid-1)
    ax.set_title(str(epochen)+' Epochen')
    ax.set_aspect('equal')
    plt.show()
    if saveFig: plt.savefig('pDirs'+str(epochen)+'.pdf')
    
    