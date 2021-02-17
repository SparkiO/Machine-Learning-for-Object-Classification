import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as backend
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout, ZeroPadding2D, BatchNormalization, AveragePooling2D, Activation, Concatenate, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import pickle
import numpy as np
import matplotlib.pyplot as pyplot

train_x = train_y = []
test_x = test_y = []

### Loads the files from the given directory. Return two sets of images and their labels.
def load_images(rootpath, set_type):
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
            if set_type != "test":
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                labels.append(label)
                image_flipped = np.flipud(image)
                images.append(image_flipped)
                labels.append(label)

    encoder = LabelEncoder()
    labels = to_categorical(encoder.fit_transform(labels))
    return np.array(images), labels

### Initalise the VGG model
def create_VGG_model():
    input_shape = Input(shape=(224, 224, 3))

    ### Block 1
    #Two Convolutions 3x3:
    conv_3x3 = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(input_shape)
    conv_3x3 = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)

    #MaxPool:
    max_pool = MaxPool2D(pool_size=(2,2), strides=(2,2))(conv_3x3)

    ### Block 2
    #Two Convolutions 3x3:
    conv_3x3 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(max_pool)
    conv_3x3 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)

    #MaxPool:
    max_pool = MaxPool2D(pool_size=(2,2), strides=(2,2))(conv_3x3)

    ### Block 3
    #Three Convolutions 3x3:
    conv_3x3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(max_pool)
    conv_3x3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)
    conv_3x3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)

    #MaxPool:
    max_pool = MaxPool2D(pool_size=(2,2), strides=(2,2))(conv_3x3)

    ### Block 4
    #Three Convolutions 3x3:
    conv_3x3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(max_pool)
    conv_3x3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)
    conv_3x3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)

    #MaxPool:
    max_pool = MaxPool2D(pool_size=(2,2), strides=(2,2))(conv_3x3)

    ### Block 5
    #Three Convolutions 3x3:
    conv_3x3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(max_pool)
    conv_3x3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)
    conv_3x3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_3x3)
    
    #MaxPool:
    max_pool = MaxPool2D(pool_size=(2,2), strides=(2,2))(conv_3x3)

    ### Output Block
    flatten = Flatten()(max_pool)
    dense_1 = Dense(4096, activation="relu")(flatten)
    dense_2 = Dense(4096, activation="relu")(dense_1)
    dense_3 = Dense(5)(dense_2)
    softmax_output = Activation("softmax")(dense_3)

    return Model(input_shape, softmax_output)

### Compile and train the model.
def compile_train(model_VGG, save_path):
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint('best_modelvgg.h5', monitor ='val_loss', save_best_only = True, save_weights_only=True)
    history = History()
    callbacks = [reduce_learning_rate, checkpoint, history]
    
    model_VGG.compile(
        optimizer = Adam(lr= 1E-4), 
        loss = "categorical_crossentropy", 
        metrics = ['accuracy']
        )

    model_VGG.fit(
        train_x, train_y,
        validation_data= (test_x, test_y),
        epochs= 16,
        batch_size = 32,
        shuffle= True,
        callbacks = callbacks
    )
    with open('trainHistoryVGG', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)   

    model_VGG.save("VGG.h5")

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
    VGG = create_VGG_model()
    compile_train(VGG, model_save_path) 

main()