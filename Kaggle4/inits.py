import os
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

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
            raise DownloadError(f"Could not find {img_dir} and downloading {img_dir + '.zip'} from the following URL failed. Please provide the data.\n{e.url}") from e

    train_datagen = ImageDataGenerator(rotation_range=30, 
                                       rescale=1./255, 
                                       width_shift_range=0.25, 
                                       height_shift_range=0.25, 
                                       shear_range=0.25, 
                                       zoom_range=0.25, 
                                       horizontal_flip=True,
                                       validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
            directory=r'./' + x_train_dir, 
            target_size=(im_size, im_size),
            color_mode='rgb', 
            batch_size=32, 
            class_mode='categorical', 
            shuffle=True, 
            subset='training')
    val_generator = train_datagen.flow_from_directory(
            directory='./' + x_train_dir,
            target_size=(im_size, im_size),
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            subset='validation')
    return train_generator, val_generator

