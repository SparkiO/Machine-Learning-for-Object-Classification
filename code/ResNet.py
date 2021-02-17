import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as backend
from tensorflow.keras.layers import GlobalAveragePooling2D, Add, Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout, ZeroPadding2D, BatchNormalization, AveragePooling2D, Activation, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.initializers import glorot_uniform

import pickle
import numpy as np
import matplotlib.pyplot as pyplot

train_x = train_y = []
test_x = test_y = []

### Loads the files from the given directory. Return two sets of images and their labels.
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
            image = np.array(load_img(f, target_size=(224, 224)), dtype="float32")
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


### ResNet
#Implementation of simple Identity Shortcut Connection block
def identity_shortcut_connection(layer, middle_shape, filters):
    filter_1, filter_2, filter_3 = filters
    input_layer = layer

    #first block component
    conv = Conv2D(filters=filter_1, kernel_size=(1,1), strides=(1,1), padding='valid')(layer)
    batch_norm = BatchNormalization(axis=3)(conv)
    relu = Activation('relu')(batch_norm)

    #second block component
    conv = Conv2D(filters=filter_2, kernel_size=(middle_shape,middle_shape), strides=(1,1), padding='same')(relu)
    batch_norm = BatchNormalization(axis=3)(conv)
    relu = Activation('relu')(batch_norm)

    #third block component
    conv = Conv2D(filters=filter_3, kernel_size=(1,1), strides=(1,1), padding='valid')(relu)
    batch_norm = BatchNormalization(axis=3)(conv)

    #add shortcut to main path
    final = Add()([batch_norm, input_layer])
    final_relu = Activation('relu')(final)

    return final_relu

#Implementation of simple Convolutional block
def convolution_block(layer, middle_shape, filters, stride=2):
    filter_1, filter_2, filter_3 = filters
    input_layer = layer

    #first block component
    conv = Conv2D(filters=filter_1, kernel_size=(1,1), strides=(stride,stride))(layer)
    batch_norm = BatchNormalization(axis=3)(conv)
    relu = Activation('relu')(batch_norm)

    #second block component
    conv = Conv2D(filters=filter_2, kernel_size=(middle_shape,middle_shape), strides=(1,1), padding='same')(relu)
    batch_norm = BatchNormalization(axis=3)(conv)
    relu = Activation('relu')(batch_norm)

    #third block component
    conv = Conv2D(filters=filter_3, kernel_size=(1,1), strides=(1,1), padding='valid')(relu)
    batch_norm = BatchNormalization(axis=3)(conv)

    #shortcut path
    conv_shortcut = Conv2D(filters=filter_3, kernel_size=(1,1), strides=(stride, stride), padding='valid')(input_layer)
    batch_norm_shortcut = BatchNormalization(axis=3)(conv_shortcut)

    #add shortcut to main path
    final = Add()([batch_norm, batch_norm_shortcut])
    final_relu = Activation('relu')(final)

    return final_relu

### Initalise the ResNet model
def create_ResNet_model():
    input_shape = Input(shape=(224, 224, 3))
    padd = ZeroPadding2D((3,3))(input_shape)

    ### Block 1
    conv = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2))(padd)
    batch_norm = BatchNormalization(axis=3)(conv)
    relu = Activation('relu')(batch_norm)
    max_pool = MaxPool2D(pool_size=(3,3), strides=(2,2))(relu)

    ### Block 2
    conv_block = convolution_block(max_pool, 3, [64,64,256])
    identity_block_1 = identity_shortcut_connection(conv_block, 3, [64,64,256])
    identity_block_2 = identity_shortcut_connection(identity_block_1, 3, [64,64,256])

    ### Block 3
    conv_block = convolution_block(identity_block_2, 3, [128,128,512])
    identity_block_1 = identity_shortcut_connection(conv_block, 3, [128,128,512])
    identity_block_2 = identity_shortcut_connection(identity_block_1, 3, [128,128,512])
    identity_block_3 = identity_shortcut_connection(identity_block_2, 3, [128,128,512])

    ### Block 4
    conv_block = convolution_block(identity_block_3, 3, [256,256,1024])
    identity_block_1 = identity_shortcut_connection(conv_block, 3, [256,256,1024])
    identity_block_2 = identity_shortcut_connection(identity_block_1, 3, [256,256,1024])
    identity_block_3 = identity_shortcut_connection(identity_block_2, 3, [256,256,1024])
    identity_block_4 = identity_shortcut_connection(identity_block_3, 3, [256,256,1024])
    identity_block_5 = identity_shortcut_connection(identity_block_4, 3, [256,256,1024])

    ### Block 5
    conv_block = convolution_block(identity_block_5, 3, [512,512,2048])
    identity_block_1 = identity_shortcut_connection(conv_block, 3, [512,512,2048])
    identity_block_2 = identity_shortcut_connection(identity_block_1, 3, [512,512,2048])

    ### AveragePool
    avg_pool = AveragePooling2D(pool_size=(2,2))(identity_block_2)

    ### Output Block
    flatten = Flatten()(avg_pool)
    dense = Dense(5)(flatten)
    softmax_output = Activation("softmax")(dense)

    return Model(input_shape, softmax_output)

### Compile and train the model.
def compile_train(model_ResNet, save_path):
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint('best_modelres.h5', monitor ='val_loss',       save_best_only = True, save_weights_only=True)
    history = History()
    callbacks = [reduce_learning_rate, checkpoint, history]
    
    model_ResNet.compile(
        optimizer = Adam(lr= 1E-3),
        loss = "categorical_crossentropy",
        metrics = ['accuracy']
        )
   

    model_ResNet.fit(
        train_x, train_y,
        validation_data= (test_x, test_y),
        epochs= 31,
        batch_size = 64,  
        shuffle = True,
        callbacks = callbacks
    )

    #save the history object if needed later
    with open('trainHistoryResNet', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)  

    model_ResNet.save("ResNet.h5")

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
    ResNet = create_ResNet_model()
    compile_train(ResNet, model_save_path)

main()
