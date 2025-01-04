# %% imports
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# %% load data generator
x_train_dir = 'TrainData6Sceneries'
x_test_dir = 'XTest6Sceneries'
url_prefix = 'https://hs-bochum.sciebo.de/s/k6i7tmPQ7JWf9H8/download?path=%2Fdata&files='

class DownloadError(Exception):
    def __init__(self, msg, url=None, filename=None):
        super().__init__(msg)
        self.url = url
        self.filename = filename

def download_and_unzip(filename):
    # check if filename exists, otherwise download it first
    if not os.path.isfile(filename):
        from urllib.request import urlretrieve
        url = url_prefix + filename
        print(f'Downloading {filename} from {url}.')
        try:
            urlretrieve(url, filename)
        except Exception as e:
            raise DownloadError('Could not Download the file', url, filename) from e

    print(f'Extracting CSV file(s) from {filename}.')
    from zipfile import ZipFile
    with ZipFile(filename, 'r') as zip_file:
        zip_file.extractall()

def load_data(im_size=150):
    # Check if CSV files exist, download and unzip otherwise
    for img_dir in (x_train_dir, x_test_dir):
        try:
            if not os.path.isdir(img_dir):
                download_and_unzip(img_dir + '.zip')
        except DownloadError as e:
            raise DownloadError(f'Could not find {img_dir} and downloading {img_dir + ".zip"} from the following URL failed. Please provide the data.\n{e.url}') from e

    train_datagen = ImageDataGenerator(rotation_range=30, rescale=1/255, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
            directory=r"./" + x_train_dir, target_size=(im_size, im_size),
            color_mode="rgb", batch_size=128, class_mode="categorical", shuffle=True)

    return train_generator

train_generator = load_data()

# %% model
# FIXME: Das ist nur ein Beispiel für ein Modell. Verbessern Sie es!
cnn = Sequential()
cnn.add(Conv2D(filters=24, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=train_generator.target_size + (3,)))
cnn.add(BatchNormalization(scale=False))
cnn.add(MaxPool2D(pool_size=(3, 3)))
cnn.add(Flatten())
cnn.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
cnn.add(Dense(6, activation='softmax'))
cnn.summary()
cnn.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(train_generator, epochs=1, verbose=True)

# %% evaluation on train set
train_generator.shuffle = False 
train_generator.index_array = None

print(cnn.evaluate(train_generator))

y_pred_train = cnn.predict(train_generator, verbose=True)
y_pred_train = np.argmax(y_pred_train, axis=1)

print('Confusion Matrix Train')
print(confusion_matrix(train_generator.classes, y_pred_train))

# %% evaluation on test set
X_test = []
for image_file in os.listdir(x_test_dir):
    image = load_img(os.path.join(x_test_dir, image_file)).resize(train_generator.target_size)
    X_test.append(img_to_array(image))

X_test = np.array(X_test) / 255

yp = cnn.predict(X_test)
yp = np.argmax(yp, axis=1)

ids = np.arange(1, yp.shape[0] + 1, dtype=np.int32).reshape(-1, 1)
combined = np.hstack((ids, yp.reshape(-1, 1)))
dt = np.dtype([('Id', np.int32),('category', np.int32)])
yp = np.array([tuple(row) for row in combined], dtype=dt)
np.savetxt("yPredict_TomTheiss.csv", yp, fmt=['%d', '%d'], delimiter=",", header="id,category", comments='')
