import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as backend
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout, ZeroPadding2D, BatchNormalization, AveragePooling2D, Activation, Concatenate, Input
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

### Initalise the GoogLeNet model
def create_GoogLeNet_model():
    input_shape = Input(shape=(224, 224, 3))

    #Convolution 7x7:
    conv_7x7 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), activation="relu", padding='same')(input_shape)

    #MaxPool:
    max_pool = MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same')(conv_7x7)

    #Convolution 1x1:
    conv_7x7x1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation="relu", padding='same')(max_pool)

    #Convolution 3x3:
    conv_3x3 = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), activation="relu", padding='same')(conv_7x7x1)

    #MaxPool:
    max_pool_3a_input = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_3x3)

    #Inception Module 3a:
    inception_1x1 = Conv2D(filters=64, kernel_size=(1,1), activation="relu", padding='same')(max_pool_3a_input)
    inception_3x3_reduce = Conv2D(filters=96, kernel_size=(1,1), activation="relu", padding='same')(max_pool_3a_input)
    inception_3x3 = Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=16, kernel_size=(1,1), activation="relu", padding='same')(max_pool_3a_input)
    inception_5x5 = Conv2D(filters=32, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(max_pool_3a_input)
    inception_max_pool_conv = Conv2D(filters=32, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_3a_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])
    
    #Inception Module 3b:
    inception_1x1 = Conv2D(filters=128, kernel_size=(1,1), activation="relu", padding='same')(inception_3a_output)
    inception_3x3_reduce = Conv2D(filters=128, kernel_size=(1,1), activation="relu", padding='same')(inception_3a_output)
    inception_3x3 = Conv2D(filters=192, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=32, kernel_size=(1,1), activation="relu", padding='same')(inception_3a_output)
    inception_5x5 = Conv2D(filters=96, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_3a_output)
    inception_max_pool_conv = Conv2D(filters=64, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_3a_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])
    
    #MaxPool:
    inception_4a_input = MaxPool2D(pool_size=(3, 3), strides=(2,2))(inception_3a_output)

    #Inception Module 4a:
    inception_1x1 = Conv2D(filters=192, kernel_size=(1,1), activation="relu", padding='same')(inception_4a_input)
    inception_3x3_reduce = Conv2D(filters=96, kernel_size=(1,1), activation="relu", padding='same')(inception_4a_input)
    inception_3x3 = Conv2D(filters=208, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=16, kernel_size=(1,1), activation="relu", padding='same')(inception_4a_input)
    inception_5x5 = Conv2D(filters=48, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_4a_input)
    inception_max_pool_conv = Conv2D(filters=64, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_4a_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])
    
    #Auxiliary classifier 1:
    class1_average_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3))(inception_4a_output)
    class1_convolution = Conv2D(filters=128, kernel_size=(1,1), activation="relu")(class1_average_pool)
    class1_flatten = Flatten()(class1_convolution)
    class1_fully_connected = Dense(1024, activation="relu")(class1_flatten)
    class1_dropout = Dropout(rate=0.7)(class1_fully_connected)
    class1_dense = Dense(5)(class1_dropout)
    class1_activation = Activation("softmax")(class1_dense)

    #Inception Module 4b:
    inception_1x1 = Conv2D(filters=160, kernel_size=(1,1), activation="relu", padding='same')(inception_4a_output)
    inception_3x3_reduce = Conv2D(filters=112, kernel_size=(1,1), activation="relu", padding='same')(inception_4a_output)
    inception_3x3 = Conv2D(filters=224, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=24, kernel_size=(1,1), activation="relu", padding='same')(inception_4a_output)
    inception_5x5 = Conv2D(filters=64, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_4a_output)
    inception_max_pool_conv = Conv2D(filters=64, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_4b_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])
    
    #Inception Module 4c:
    inception_1x1 = Conv2D(filters=128, kernel_size=(1,1), activation="relu", padding='same')(inception_4b_output)
    inception_3x3_reduce = Conv2D(filters=128, kernel_size=(1,1), activation="relu", padding='same')(inception_4b_output)
    inception_3x3 = Conv2D(filters=256, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=24, kernel_size=(1,1), activation="relu", padding='same')(inception_4b_output)
    inception_5x5 = Conv2D(filters=64, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_4b_output)
    inception_max_pool_conv = Conv2D(filters=64, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_4c_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])

    #Inception Module 4d:
    inception_1x1 = Conv2D(filters=112, kernel_size=(1,1), activation="relu", padding='same')(inception_4c_output)
    inception_3x3_reduce = Conv2D(filters=144, kernel_size=(1,1), activation="relu", padding='same')(inception_4c_output)
    inception_3x3 = Conv2D(filters=288, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=32, kernel_size=(1,1), activation="relu", padding='same')(inception_4c_output)
    inception_5x5 = Conv2D(filters=64, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_4c_output)
    inception_max_pool_conv = Conv2D(filters=64, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_4d_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])
    
    #Auxiliary classifier 2:
    class2_average_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3))(inception_4d_output)
    class2_convolution = Conv2D(filters=128, kernel_size=(1,1), activation="relu")(class2_average_pool)
    class2_flatten = Flatten()(class2_convolution)
    class2_fully_connected = Dense(1024, activation="relu")(class2_flatten)
    class2_dropout = Dropout(rate=0.7)(class2_fully_connected)
    class2_dense = Dense(5)(class2_dropout)
    class2_activation = Activation("softmax")(class2_dense)

    #Inception Module 4e:
    inception_1x1 = Conv2D(filters=256, kernel_size=(1,1), activation="relu", padding='same')(inception_4d_output)
    inception_3x3_reduce = Conv2D(filters=160, kernel_size=(1,1), activation="relu", padding='same')(inception_4d_output)
    inception_3x3 = Conv2D(filters=320, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=32, kernel_size=(1,1), activation="relu", padding='same')(inception_4d_output)
    inception_5x5 = Conv2D(filters=128, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_4d_output)
    inception_max_pool_conv = Conv2D(filters=128, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_4e_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])

    #MaxPool:
    inception_5a_input = MaxPool2D(pool_size=(3, 3), strides=(2,2))(inception_4e_output)

    #Inception Module 5a:
    inception_1x1 = Conv2D(filters=256, kernel_size=(1,1), activation="relu", padding='same')(inception_5a_input)
    inception_3x3_reduce = Conv2D(filters=160, kernel_size=(1,1), activation="relu", padding='same')(inception_5a_input)
    inception_3x3 = Conv2D(filters=320, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=32, kernel_size=(1,1), activation="relu", padding='same')(inception_5a_input)
    inception_5x5 = Conv2D(filters=128, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_5a_input)
    inception_max_pool_conv = Conv2D(filters=128, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_5a_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])

    #Inception Module 5b:
    inception_1x1 = Conv2D(filters=384, kernel_size=(1,1), activation="relu", padding='same')(inception_5a_output)
    inception_3x3_reduce = Conv2D(filters=192, kernel_size=(1,1), activation="relu", padding='same')(inception_5a_output)
    inception_3x3 = Conv2D(filters=384, kernel_size=(3,3), activation="relu", padding='same')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(filters=48, kernel_size=(1,1), activation="relu", padding='same')(inception_5a_output)
    inception_5x5 = Conv2D(filters=128, kernel_size=(5,5), activation="relu", padding='same')(inception_5x5_reduce)
    inception_max_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(inception_5a_output)
    inception_max_pool_conv = Conv2D(filters=128, kernel_size=(1,1), activation="relu", padding='same')(inception_max_pool)
    inception_5b_output = Concatenate(axis=3)([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_conv])

    #Final Classifier
    final_average_pool = GlobalAveragePooling2D()(inception_5b_output)
    final_dropout = Dropout(rate=0.40)(final_average_pool)
    class_final = Dense(5)(final_dropout)
    class_final_activation = Activation("softmax")(class_final)

    return Model(input_shape, [class1_activation, class2_activation, class_final_activation])

### Compile and train the model.
def compile_train(model_GoogLeNet, save_path):
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint('best_modelgoogle.h5', monitor ='val_loss', save_best_only = True, save_weights_only=True)
    history = History()
    callbacks = [reduce_learning_rate, checkpoint, history]
    
    model_GoogLeNet.compile(
        optimizer = Adam(lr= 1E-3), 
        loss = "categorical_crossentropy", 
        metrics = ['accuracy']
        )
    

    # three train_y and test_y - one for each auxiliary classifier
    model_GoogLeNet.fit(
        train_x, [train_y, train_y, train_y], 
        validation_data = (test_x, [test_y, test_y, test_y]), 
        epochs = 31,
        batch_size = 64,    
        shuffle= True,
        callbacks = callbacks
    )

    #save the history object if needed later
    with open('trainHistoryGoogleNet', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)  

    model_GoogLeNet.save("GoogLeNet.h5")

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
    GoogLeNet = create_GoogLeNet_model()
    compile_train(GoogLeNet, model_save_path) 

main()
