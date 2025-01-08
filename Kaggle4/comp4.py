# %% imports
import os
import numpy as np
from sklearn.metrics import confusion_matrix
# from keras.utils import load_img
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from inits import load_data, x_test_dir

train_generator, val_generator = load_data(256)

# %% model
# model

l2Reg = 5e-5
stumpVGG16 = VGG16(weights='imagenet', 
                   include_top=False,
                   input_shape=train_generator.target_size + (3,),
                   classes=6)
stumpVGG16.trainable=False

cnn = Sequential()
cnn.add(BatchNormalization())
cnn.add(stumpVGG16)
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2Reg)))
cnn.add(Dropout(0.5))
cnn.add(Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2Reg)))
cnn.add(Dropout(0.5))
cnn.add(Dense(6, activation='softmax'))

cnn.compile(optimizer=Adam(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy', 'precision', 'recall'])
es = EarlyStopping( monitor='val_loss', patience=7, restore_best_weights=True )
cnn.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[es], verbose=True)


for layer in stumpVGG16.layers[-4:]: layer.trainable = True
cnn.compile(optimizer=Adam(learning_rate=1e-4),loss='categorical_crossentropy',metrics=['accuracy', 'precision', 'recall'])
cnn.fit(train_generator, epochs=10, validation_data=val_generator, verbose=True)

cnn.save_weights('TomTheiss.weights.h5')
cnn.save("model.keras")
cnn.summary()


# %% evaluation on test set
# test evaluation
cnn = load_model('model.keras')
cnn.load_weights('TomTheiss.weights.h5')
X_test = []
for image_file in sorted(os.listdir(x_test_dir)):
    image = load_img(os.path.join(x_test_dir, image_file)).resize(train_generator.target_size)
    X_test.append(img_to_array(image))

X_test = np.array(X_test) / 255
yp_test = cnn.predict(X_test)
yp_test = np.argmax(yp_test, axis=1)

ids = np.arange(1, yp_test.shape[0] + 1, dtype=np.int32).reshape(-1, 1)
combined = np.hstack((ids, yp_test.reshape(-1, 1)))
dt = np.dtype([('Id', np.int32),('category', np.int32)])
yp_test = np.array([tuple(row) for row in combined], dtype=dt)
np.savetxt("yPredict_TomTheiss.csv", yp_test, fmt=['%d', '%d'], delimiter=",", header="id,category", comments='')

# %% evaluation on train set
# train evaluation
cnn = load_model('model.keras')
# cnn = load_model('model_transfer.keras')

train_generator.shuffle = False 
train_generator.index_array = None

score = cnn.evaluate(train_generator)
print("Accuracy train:\t%.2f%%" % (score[1]*100))

print('Confusion Matrix Train')
yp_train = cnn.predict(train_generator, verbose=True)
yp_train = np.argmax(yp_train, axis=1)
print(confusion_matrix(train_generator.classes, yp_train))

# %% 
# validation set
cnn = load_model('model.keras')
# cnn = load_model('model_transfer.keras')

val_generator.shuffle = False
val_generator.index_array = None

score = cnn.evaluate(val_generator)
print("Accuracy val:\t%.2f%%" % (score[1]*100))

print('Confusion Matrix Validation')
yp_val = cnn.predict(val_generator, verbose=True)
yp_val = np.argmax(yp_val, axis=1)
print(confusion_matrix(val_generator.classes, yp_val))
# %%
