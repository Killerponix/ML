import numpy as np
class TwoMoons:
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