from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as backend
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, ZeroPadding2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam

import pickle
import numpy as np
import os
import matplotlib.pyplot as pyplot
from time import time

train_x = train_y = []
test_x = test_y = []


### Loads the files from the train and test directories.
def load_images(rootpath, set_name):
    directories = [d for d in os.listdir(rootpath)
                   if os.path.isdir(os.path.join(rootpath, d))]
    labels = []
    images = []
    for label in directories:
        path = os.path.join(rootpath, label)
        file_names = [os.path.join(path, f)
                      for f in os.listdir(path)]
        for f in file_names:
            image = np.array(load_img(f, target_size=(224, 224)), dtype='float32')
            image /= 255
            images.append(image)
            labels.append(label)
            if set_name != "test":
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                labels.append(label)
                image_flipped = np.flipud(image)
                images.append(image_flipped)
                labels.append(label)

    encoder = LabelEncoder()
    labels = to_categorical(encoder.fit_transform(labels))
    return np.array(images), labels


### Initalise the ZFNet model
def create_ZFNet_model():
    model_ZFNet = Sequential([
        #1st convolutional layer
		#Compared to AlexNet - reduced filter from 11 x 11 to 7 x 7 and strides from 4 x 4  to 2 x 2
        Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), input_shape=(224,224,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2,2)),

        #2nd convolutional layer
        Conv2D(256, kernel_size=(5,5), strides=(2,2)),
        Activation('relu'),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        #3rd convolution
        Conv2D(384, kernel_size=(3, 3), strides=(1, 1)),
        Activation('relu'),

        #4th convolution
        Conv2D(384, kernel_size=(3, 3), strides=(1, 1)),
        Activation('relu'),

        #5th convolution
        Conv2D(256, kernel_size=(3, 3), strides=(1, 1)),
        Activation('relu'),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        #1st fully connected Layer
        Flatten(),
        Dense(4096, input_shape=(224*224*3,)),
        Activation('relu'),
        Dropout(0.4),
        
        #2nd fully connected layer
        Dense(4096),
        Activation('relu'),
        Dropout(0.4),

        #3rd fully connected layer
        Dense(1000),
        Activation('relu'),
        Dropout(0.4),

        #output
        Dense(5),
        Activation('softmax')
    ])

    return model_ZFNet

### Compile and train the model.
def compile_train(model_ZFNet, save_path):
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint('best_modelZFNet.h5', monitor ='val_loss', save_best_only = True, save_weights_only=True)
    history = History()
    callbacks = [reduce_learning_rate, checkpoint, history]
    
    model_ZFNet.compile(
        optimizer = Adam(lr= 1E-3), 
        loss = "categorical_crossentropy", 
        metrics= ['accuracy']
        )
    

    model_ZFNet.fit(
        train_x, train_y,
        validation_data= (test_x, test_y),
        epochs= 31,
        batch_size = 64,  
        shuffle=True,
        callbacks = callbacks
    )

    model_ZFNet.save("ZFNet.h5")

    #save the history object if needed later
    with open('trainHistoryZFNet', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)  


def main():
    global train_x
    global train_y
    global test_x
    global test_y
    train_path = "data/train"
    test_path = "data/test"
    model_save_path = "data/"

    train_x, train_y = load_images(train_path, "train")
    test_x, test_y = load_images(test_path, "test")
    ZFNet = create_ZFNet_model()
    compile_train(ZFNet, model_save_path) 

main()