import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, BatchNormalization
from keras.optimizers import Adam
import keras
# from tensorflow.compat.v1.random import set_random_seed
# import tensorflow as tf

import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)
keras.utils.set_random_seed(42)

# shape:  (24000, 2000) 
# -> 8000 Signale, je 3 Kanäle
# -> 2000 Daten pro Kanal
XTrain = pd.read_csv('XTrainSignale.csv', delimiter=' ', header=None) 
XTest  = pd.read_csv('XTestSignale.csv',  delimiter=' ', header=None) # shape:  (6000, 2000)
YTrain = pd.read_csv('YTrainSignale.csv', delimiter=' ', header=None) # shape:  (8000, 1)

XTrain = XTrain.values.reshape(XTrain.shape[0]//3, XTrain.shape[1], 3)
XTest  = XTest.values.reshape(  XTest.shape[0]//3,  XTest.shape[1], 3)

YTrain = YTrain.values.flatten()

# Daten normalisieren
XTrain = XTrain / np.max(XTrain)
XTest = XTest / np.max(XTest)

TrainSet = np.random.choice(XTrain.shape[0],int(XTrain.shape[0]*0.7), replace=False)
ValSet   = np.delete(np.arange(0, XTrain.shape[0] ), TrainSet) 
XVal     = XTrain[ValSet,:]
YVal     = YTrain[ValSet]
XTrain   = XTrain[TrainSet,:]
YTrain   = YTrain[TrainSet]

myCNN = Sequential()
myCNN.add(Conv1D( 3, kernel_size=21, activation='relu', use_bias=False, input_shape=(XTrain.shape[1],XTrain.shape[2])))
#myCNN.add(MaxPooling1D(pool_size=2))
myCNN.add(Conv1D( 5, kernel_size=7, activation='relu', use_bias=False))
#myCNN.add(MaxPooling1D(pool_size=2))
myCNN.add(Flatten())
myCNN.add(BatchNormalization())
#myCNN.add(Dropout(0.3))
myCNN.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
myCNN.add(BatchNormalization())
#myCNN.add(Dropout(0.3))
myCNN.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
myCNN.add(Dense(4,  activation='softmax', kernel_initializer='glorot_uniform'))
myCNN.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint('bestW.weights.h5', monitor='val_accuracy', mode='max', verbose=False, save_weights_only=True, save_best_only=True)

history = myCNN.fit(XTrain, YTrain, epochs=70, validation_data=(XVal, YVal), callbacks=[checkpoint], verbose=True, batch_size=32)
myCNN.summary()

myCNN.load_weights('bestW.weights.h5')

yp = myCNN.predict(XTest)

val_acc = history.history['val_accuracy']
max_val_acc = max(val_acc)
best_epochs = [i + 1 for i, acc in enumerate(val_acc) if acc == max_val_acc]

print(f"Best Val-Acc :\t{max(val_acc):.4f}\nat epochs :\t{best_epochs}")

yp = np.argmax(yp, axis=1)

ids = np.arange(1, yp.shape[0] + 1, dtype=np.int32).reshape(-1, 1)
combined = np.hstack((ids, yp.reshape(-1, 1)))
dt = np.dtype([('Id', np.int32),('category', np.int32)])
yp = np.array([tuple(row) for row in combined], dtype=dt)
np.savetxt("yPredict_TomTheiss.csv", yp, fmt=['%d', '%d'], delimiter=",", header="id,category", comments='')