import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

class qFunctionANN:
    def __init__(self,stateDim,actions, networkArc=[100], l2Reg=0.001, 
                 activation = 'tanh', tablesize=[19,19]):
        self._stateDim = stateDim
        self.unfitted = True
        self._QFunction = Sequential()
        self._QFunction.add( Dense(networkArc[0], kernel_regularizer=l2(l2Reg), 
                                   input_dim=stateDim+1, activation=activation) )
        self._QFunction.add(tf.keras.layers.BatchNormalization())
        for i in range(1,len(networkArc)):
            self._QFunction.add(Dense(networkArc[i],  kernel_regularizer=l2(l2Reg), 
                                      activation=activation))
            self._QFunction.add(tf.keras.layers.BatchNormalization())
        self._QFunction.add(Dense(1,  kernel_regularizer=l2(l2Reg), activation='linear')) 
        self._QFunction.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self._QFunction.summary()
        
    def fit(self,state,action,Y, epochs = 5000):
        X = np.hstack((state,action))
        earlystop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, 
                                                      verbose=False,restore_best_weights=True)
        callbacksList = [earlystop] 
        ValSet    = np.random.choice(X.shape[0],int(X.shape[0]*0.2),replace=False)
        TrainSet  = np.delete(np.arange(0, Y.shape[0] ), ValSet) 
        XVal     = X[ValSet,:]
        YVal     = Y[ValSet]
        X        = X[TrainSet,:]
        Y        = Y[TrainSet]
        history = self._QFunction.fit(X,Y,epochs=epochs, validation_data=(XVal, YVal), 
                                      callbacks=callbacksList ,verbose=False,batch_size=1000) 
        self.unfitted = False
    
    def predict(self,state,action):
        X = np.hstack((state,action))
        if self.unfitted:
            if len(X.shape) == 1: Qsa = np.array([0])
            else: Qsa = np.zeros((state.shape[0],1))
        else:
            if len(X.shape) == 1: X = X[np.newaxis,:]
            Qsa = self._QFunction.predict(X)
        tf.keras.backend.clear_session() #*\label{code:agent:qf:0}
        return Qsa

