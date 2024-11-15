import numpy as np

MainSet = np.arange(0,12)
Set1    = np.random.choice(12,4, replace=False)
Set2    = np.delete(MainSet, Set1) 
