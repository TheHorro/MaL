# %% Data
# Load Test-Data
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)

def divValTrainSet(XTrain,yTrain):
    ValSet    = np.random.choice(XTrain.shape[0],int(XTrain.shape[0]*0.2),replace=False)
    TrainSet  = np.delete(np.arange(0, yTrain.shape[0] ), ValSet) 
    XVal     = XTrain[ValSet,:]
    yVal     = yTrain[ValSet]
    X        = XTrain[TrainSet,:]
    y        = yTrain[TrainSet]
    return (XVal, yVal, X, y)

#Daten Laden
# Load the fashion-mnist pre-shuffled train data and test data
(XTrain, yTrain), (XTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()

##Daten kommen optimiert fuer klassische neuronale Netze, daher wieder in Bildformat bringen
img_rows, img_cols = 28, 28
XTrain = XTrain.reshape(XTrain.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0
XTest = XTest.reshape(XTest.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0

yTrain = to_categorical(yTrain)
yTest = to_categorical(yTest)

XVal, yVal, XTrain, yTrain = divValTrainSet(XTrain, yTrain)

# %% 
# Model
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

l2Reg = 5e-5
cnn = Sequential()
cnn.add(BatchNormalization())
cnn.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2Reg), input_shape=(28,28,1)))
cnn.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
# cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2Reg)))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2Reg)))
#cnn.add(Dropout(0.5))
cnn.add(Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2Reg)))
#cnn.add(Dropout(0.5))
cnn.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2Reg)))
#cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
earlystop = EarlyStopping( monitor='val_loss', patience=4, restore_best_weights=True )
checkpoint = ModelCheckpoint('bestW.weights.h5', monitor='val_loss', verbose=False, save_weights_only=True, save_best_only=True)
# cnn.fit( XTrain, yTrain, epochs=20, validation_data=(XVal, yVal), callbacks=[earlystop, checkpoint], verbose=True)
# cnn.save("model.keras")
cnn.summary()

# %% evaluation
# evaluation
cnn = load_model('model.keras')
cnn.load_weights('bestW.weights.h5')

def eval(X, y, label:str):
	score = cnn.evaluate(X, y, verbose=False)
	print("Accuracy %s:\t%.2f%%" % (label, score[1]*100))
	yp = cnn.predict(X, verbose=False)
	yp = np.argmax(yp, axis=1)
	y_labels = np.argmax(y, axis=1)
	print(f'Confusion Matrix {label}')
	mat = np.zeros((10,10), dtype=(np.int32))
	for i in range(y_labels.shape[0]): mat[y_labels[i],yp[i]] += 1
	print(mat)

eval(XTrain, yTrain, 'train')
eval(XVal, yVal, 'validation')
eval(XTest, yTest, 'test')

yp_test = cnn.predict(XTest)
yp_test = np.argmax(yp_test, axis=1)
yTest_labels = np.argmax(yTest, axis=1)

yp_test = yp_test.reshape(yp_test.shape[0])
error = np.count_nonzero(yp_test - yTest_labels)
print(f'Fehler: {error}/{yp_test.shape[0]}')

# %%
