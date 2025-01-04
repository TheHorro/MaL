import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["OMP_NUM_THREADS"] = "16"  # Entspricht der Anzahl der logischen Kerne
os.environ["TF_NUM_INTRA_OP_THREADS"] = "16"
os.environ["TF_NUM_INTER_OP_THREADS"] = "16"


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(42)

XTest = np.load('XTest.npy')
XTrain = np.load('XTrain.npy')
YTrain = np.load('YTrain.npy')
YTest = np.load('YTest.npy')

def divValTrainSet(X,Y):
    ValSet    = np.random.choice(X.shape[0],int(X.shape[0]*0.25),replace=False)
    TrainSet  = np.delete(np.arange(0, Y.shape[0] ), ValSet) 
    XVal     = X[ValSet,:]
    YVal     = Y[ValSet]
    X        = X[TrainSet,:]
    Y        = Y[TrainSet]
    return (XVal, YVal, X, Y)

(XVal, YVal, XTr, YTr) = divValTrainSet(XTrain,YTrain)

myANN = Sequential()
# 100, softplus, glorot_normal -> 34.39813896049996
myANN.add(Dense(100, activation='softplus', kernel_initializer='glorot_normal', use_bias=True, input_dim=XTr.shape[1]))
myANN.add(Dense(100, activation='softplus', kernel_initializer='glorot_normal', use_bias=True))
myANN.add(Dense(100, activation='softplus', kernel_initializer='glorot_normal', use_bias=True))
myANN.add(Dense(100, activation='softplus', kernel_initializer='glorot_normal', use_bias=True))
myANN.add(Dense(  1,   activation='linear', kernel_initializer='glorot_normal', use_bias=True))
myANN.compile(loss='mean_squared_error', optimizer='adam')

earlystop  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=False, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('bestW.weights.h5', monitor='val_loss', verbose=False, save_weights_only=True, save_best_only=True)
callbacksList = [checkpoint, earlystop] 

history = myANN.fit(XTr,YTr, epochs=2000, validation_data=(XVal, YVal), callbacks=callbacksList, verbose=False)
myANN.load_weights('bestW.weights.h5')

yp = myANN.predict(XTest)
# ids = np.arange(1, yp.shape[0] + 1, dtype=np.int32).reshape(-1, 1)
# combined = np.hstack((ids, yp.reshape(-1, 1)))
# 
# dt = np.dtype([('Id', np.int32),('Predict', np.float32)])
# yp = np.array([tuple(row) for row in combined], dtype=dt)
# np.savetxt("yPredict_TomTheiss.csv", yp, delimiter=",", fmt=['%d', '%.15f'], header="Id,Predicted", comments='')


yp = yp.reshape(yp.shape[0])
error = np.mean(np.abs(yp - YTest))
print('Fehler: ', error)