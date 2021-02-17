import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as backend
from tensorflow.keras.layers import DepthwiseConv2D, GlobalMaxPooling2D, Add, ReLU, Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout, ZeroPadding2D, BatchNormalization, AveragePooling2D, Activation, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.initializers import glorot_uniform

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
            image = np.array(load_img(f, target_size=(224, 224)))
            image = image.astype("float32")
            image /= 255
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


### MNASNet
#Implementation of inverted residual block
def inverted_residual_block(layer, filter_size, exp, filters, stride=1):
    
    conv = Conv2D(exp*layer.shape[1], kernel_size=(1,1), strides=(stride,stride), padding="same")(layer)
    batch_norm = BatchNormalization(epsilon=1e-3)(conv)
    relu = ReLU(6.)(batch_norm)

    conv_depth = DepthwiseConv2D(kernel_size=filter_size, strides=stride, padding='same')(relu)
    batch_norm = BatchNormalization(epsilon=1e-3)(conv_depth)
    relu = ReLU(6.)(batch_norm)

    conv = Conv2D(filters=filters, kernel_size=(1,1), padding='same')(relu)
    batch_norm = BatchNormalization(epsilon=1e-3)(conv)

    return batch_norm


### Initalise the MNASNet model
def create_MNASNet_model(depth_multi=1):
    input_shape = Input(shape=(224, 224, 3))

    ### Conv 3x3 layer block:
    conv = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2))(input_shape)
    batch_norm = BatchNormalization(epsilon=1e-3)(conv)
    relu = ReLU(6.)(batch_norm)

    ### SepConv 3x3 layer block:
    conv_depth = DepthwiseConv2D(kernel_size=(3,3), padding="same", strides=(1,1))(relu)
    conv = Conv2D(filters=16, kernel_size=(1,1), padding="valid", strides=(1,1))(conv_depth)
    batch_norm = BatchNormalization(epsilon=1e-3)(conv)
    relu = ReLU(6.)(batch_norm)

    ### Inverted Blocks 1:
    inv_1 = inverted_residual_block(relu, filter_size=3, exp=3, stride=2, filters=24)
    inv_2 = inverted_residual_block(inv_1, filter_size=3, exp=3, stride=1, filters=24)
    inv_3 = inverted_residual_block(inv_2, filter_size=3, exp=3, stride=1, filters=24)

    ### Inverted Blocks 2:
    inv_4 = inverted_residual_block(inv_3, filter_size=5, exp=3, stride=2, filters=40)
    inv_5 = inverted_residual_block(inv_4, filter_size=5, exp=3, stride=1, filters=40)
    inv_6 = inverted_residual_block(inv_5, filter_size=5, exp=3, stride=1, filters=40)

    ### Inverted Blocks 3:
    inv_7 = inverted_residual_block(inv_6, filter_size=5, exp=6, stride=2, filters=80)
    inv_8 = inverted_residual_block(inv_7, filter_size=5, exp=6, stride=1, filters=80)
    inv_9 = inverted_residual_block(inv_8, filter_size=5, exp=6, stride=1, filters=80)

    ### Inverted Blocks 4:
    inv_10 = inverted_residual_block(inv_9, filter_size=3, exp=6, stride=1, filters=96)
    inv_11 = inverted_residual_block(inv_10, filter_size=3, exp=6, stride=1, filters=96)

    ### Inverted Blocks 5:
    inv_12 = inverted_residual_block(inv_11, filter_size=5, exp=6, stride=2, filters=192)
    inv_13 = inverted_residual_block(inv_12, filter_size=5, exp=6, stride=1, filters=192)
    inv_14 = inverted_residual_block(inv_13, filter_size=5, exp=6, stride=1, filters=192)
    inv_15 = inverted_residual_block(inv_14, filter_size=5, exp=6, stride=1, filters=192)

    ### Inverted Blocks 6:
    inv_final = inverted_residual_block(inv_15, filter_size=3, exp=6, stride=1, filters=320)

    ### AveragePool:
    avg_pool = GlobalMaxPooling2D()(inv_final)

    ### Output Block:
    dense = Dense(5)(avg_pool)
    softmax_output = Activation("softmax")(dense)

    return Model(input_shape, softmax_output)

### Compile and train the model.
def compile_train(model_MNASNet, save_path):
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint('best_modelMNasNet.h5', monitor ='val_loss', save_best_only = True, save_weights_only=True)
    history = History()
    callbacks = [reduce_learning_rate, checkpoint, history]
    
    model_MNASNet.compile(
        optimizer = Adam(lr=1e-5),
        loss = "categorical_crossentropy",
        metrics = ['accuracy']
        )
   

    model_MNASNet.fit(
        train_x, train_y,
        validation_data= (test_x, test_y),
        epochs= 41,
        batch_size = 64,  
        callbacks = callbacks
    )

    #save the history object if needed later
    with open('trainHistoryMNASNet', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)  

    model_MNASNet.save("MNASNet.h5")
    plotResults(history)

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
    MNASNet = create_MNASNet_model()
    MNASNet.summary()
    compile_train(MNASNet, model_save_path)

main()