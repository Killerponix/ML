import numpy as np
from sklearn.tree import DecisionTreeRegressor

class gradientBoost:
    def __init__(self,noOfLearner=10, maxDepth =4, eta = 0.2):
        self.noOfLearner =  noOfLearner
        self.maxDepth = maxDepth
        self.eta = eta
        self.learner = []
        self.startValue = None
        
    def fit(self,X,y):
        self.startValue = np.mean(y)
        yP = self.startValue * np.ones(X.shape[0])
             
        for i in range(0,self.noOfLearner):
            r = -(yP - y)
            learner = DecisionTreeRegressor(max_depth=self.maxDepth)
            learner.fit(X,r)
            self.learner.append(learner)
            yP += self.eta * learner.predict(X)
            
    def predict(self,X, level=np.inf):
        y = self.startValue * np.ones(X.shape[0])
        for count, learner in enumerate(self.learner):
            if count>level-1: break
            y += self.eta * learner.predict(X)
        return(y)
        
if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    np.random.seed(42)
    X = np.linspace(0,1,1000)
    yT = np.round(3*np.cos(2*np.pi*X)*np.sin(np.pi*X+0.2))
    y  = yT + 0.2*(np.random.rand(1000)-0.5)
    X  = X.reshape(1000,1)   
    gb = gradientBoost(noOfLearner=20, eta=0.5, maxDepth = 2)
    gb.fit(X,y)
    errorList = []
    for i in range(20):
        yP = gb.predict(X,level=i)
        plt.figure()
        plt.scatter(X,y,c='r', alpha=0.15)
        plt.plot(X,yP,c='k',lw=2)
        plt.ylim([-3.1,1.1])
        plt.title('Level '+str(i))
        errorList.append(np.mean(np.abs(yP - yT)))
        
    plt.figure()
    plt.plot(errorList,c='k',lw=2, marker='*', markersize=10)
    plt.title('Entwicklung des Fehlers')
    plt.xlabel('Anzahl Lerner')
    plt.ylabel('Mittlerer Absoluter Fehler')

    f = open("hourCleanUp.csv") 
    header = f.readline().rstrip('\n')  
    featureNames = header.split(',')
    dataset = np.loadtxt(f, delimiter=",")
    f.close()   
    X = dataset[:,0:13]
    Y = dataset[:,15]   
    index = np.flatnonzero(X[:,8]==4)
    X = np.delete(X,index, axis=0)
    Y = np.delete(Y,index, axis=0)   
    np.random.seed(42)
    MainSet = np.arange(0,X.shape[0])
    Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
    Testset = np.delete(MainSet,Trainingsset)
    XTrain = X[Trainingsset,:]
    yTrain = Y[Trainingsset]
    XTest = X[Testset,:]
    yTest = Y[Testset] 
    import time
    
    gb = gradientBoost(noOfLearner=100,eta=0.1,maxDepth=12)  
#    gb = gradientBoost(noOfLearner=500,eta=0.2,maxDepth=5)  
    startProc = time.process_time()
    gb.fit(XTrain,yTrain)
    yPredict = np.round(gb.predict(XTest))
    endeProc = time.process_time()
    print('Systemzeit: %5.3f s' % (endeProc-startProc))
    yDiff = yPredict - yTest
    print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
    
    from xgboost import XGBRegressor
    model = XGBRegressor(learning_rate = 0.1, max_depth = 12, n_estimators = 100)
    model.fit(XTrain,yTrain)
    yPredict = np.round(model.predict(XTest))
    yDiff = yPredict - yTest
    print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
    
    TrainSet = np.arange(0,len(yTrain))
    Valset = np.random.choice(len(yTrain), int(0.2*len(yTrain)), replace=False)
    Trainingsset = np.delete(TrainSet,Valset)
    XVal = XTrain[Valset,:]
    yVal = yTrain[Valset]
    XTrain = XTrain[Trainingsset,:]
    yTrain = yTrain[Trainingsset]   
    model = XGBRegressor(learning_rate = 0.1, max_depth = 12, n_estimators = 500)
    model.fit(XTrain,yTrain, eval_metric='mae', early_stopping_rounds=20 , eval_set=[(XVal,yVal)], verbose=True)
    yPredict = np.round(model.predict(XTest))
    yDiff = yPredict - yTest
    print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
    