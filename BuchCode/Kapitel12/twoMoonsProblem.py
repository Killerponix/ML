import numpy as np

def twoMoonsProblem( SamplesPerMoon=240, pNoise=2):

    tMoon0 = np.linspace(0, np.pi, SamplesPerMoon)
    tMoon1 = np.linspace(0, np.pi, SamplesPerMoon)
    Moon0x = np.cos(tMoon0)
    Moon0y = np.sin(tMoon0)
    Moon1x = 1 - np.cos(tMoon1)
    Moon1y = 0.5 - np.sin(tMoon1) 

    X = np.vstack((np.append(Moon0x, Moon1x), np.append(Moon0y, Moon1y))).T
    X = X + pNoise/100*np.random.normal(size=X.shape)
    Y = np.hstack([np.zeros(SamplesPerMoon), np.ones(SamplesPerMoon)])

    return X, Y

if __name__ == '__main__':
    
    (X,Y) = twoMoonsProblem()
    
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    indexA = np.flatnonzero(Y>0.5)
    indexB = np.flatnonzero(Y<0.5)
    ax.scatter(X[indexA,0],X[indexA,1],color='red', marker='o')
    ax.scatter(X[indexB,0],X[indexB,1],color='black', marker='+')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_ylim([-1,2])
    ax.set_ylim([-1,2])
    ax.set_title("Two Moons Set")