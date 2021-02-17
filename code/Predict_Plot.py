import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np
import os

import sklearn
from sklearn.metrics import confusion_matrix

image_width, image_height = 224, 224
images_path = 'data/test/'
model_path = 'model/ResNet.h5'
labels = ["berry", "bird", "dog", "flower", "other"]

### Imports images from a folder and adds to a list
def create_list_images(path):
        images_list = []
        for img in os.listdir(path):
                img = os.path.join(path, img)
                img = image.load_img(
                        img, 
                        target_size=(image_width, image_height))

                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                images_list.append(img)
        
        return images_list

### Loads the pre-trained model and compiles it
def load_compiler_model(model_name = ""):
        model = load_model(model_path)
        # #model.compile(loss='categorical_crossentropy',
        #         optimizer='adam',
        #         metrics=['accuracy'])
        
        return model

### Takes a list of images and a model and predicts the classes of the images.
### Calls the plot function to visualise the predictions.
def predict(images_list, model):
        images = np.vstack(images_list)
        #images /= 255
        images_to_plot = []
        classes_to_plot = []
        for i in range(10):
                single_image = images[i]
                single_image = np.expand_dims(single_image, axis=0)
                classes = model.predict(single_image)
                images_to_plot.append(single_image)
                classes_to_plot.append(classes)

        plot(images_to_plot, classes_to_plot)


######### GoogLeNet version, there are 3 classifiers returning predictions #########
### Takes a list of images and a model and predicts the classes of the images.
### Calls the plot function to visualise the predictions.
def predict_GoogLeNet(images_list, model):
        images = np.vstack(images_list)
        images_to_plot = []
        classes_to_plot = []

        for i in range(10):
                prediction_three_classifiers = []

                single_image = images[i]
                single_image = np.expand_dims(single_image, axis=0)
                images_to_plot.append(single_image)

                classes = model.predict(single_image)
                print(classes)
                prediction_three_classifiers.append(classes)
                classes_to_plot.append(calculate_average(prediction_three_classifiers))
        
        plot(images_to_plot, classes_to_plot)

### For GoogLeNet - takes three vectors of probability distributions and returns one with average.
def calculate_average(three_classifiers):
        average_vec = []
        b = []
        for i in range (5):
                sum = 0
                for j in range (3):
                        b = three_classifiers[0][j].tolist()
                        sum = sum + b[0][i]

                average = sum / 3
                average_vec.append(average*100)
        
        return average_vec

### For 10 first images creates a plot with the image and corresponding
### probabilities of class distributions. Creates 5x4 grid.
def plot(img, classes):
        fig, ax = plt.subplots(5, 4)
        fig.set_size_inches(13, 10)
        plt.tight_layout()
        y_pos = np.arange(len(labels))

        for i in range(10):
                img[i] = np.squeeze(img[i], axis=0)
                img[i] /= 255
                classes[i] *= 100
                classes[i] = np.around(classes[i], 2)

                a = classes[i]
                print(a)
                b = []
                for j in range(5):
                        b.append(a[0][j])
                
                #Every second iteration changes the row
                if i % 2 is 0:
                        ax[i//2, 0].imshow(img[i],  aspect="auto", interpolation="nearest")
                        ax[i//2, 1].bar(y_pos, b, color="green", tick_label=labels)
                        ax[i//2, 1].set_ylim([0, 100])
                else:
                        ax[i//2, 2].imshow(img[i], aspect="auto", interpolation="nearest")
                        ax[i//2, 3].bar(y_pos, b, color="green", tick_label=labels)
                        ax[i//2, 3].set_ylim([0, 100])

        plt.show()

### Prepares the data for the confusion matrix.
def prepare_confusion_matrix(model):
        predicted_classes = []
        true_classes = []

        for c in labels:
                path = images_path + c
                images = create_list_images(path)
                images = np.vstack(images)
                #images /= 255
                for single_image in images:
                        single_image = np.expand_dims(single_image, axis=0)
                        predicted_class = model.predict(single_image)
                        
                        if (model_path != "model/GoogLeNet.h5"):
                                predicted_class = predicted_class.argmax(axis=-1)
                        
                        predicted_classes.append(predicted_class)
                        true_classes.append(labels.index(c))
                        
        if model_path == "model/GoogLeNet.h5":
                # Calculates the matrix for each of the 3 classifiers of GoogLeNet
                # for j in range(3):
                #         separated_classifier = []
                #         for i in range(2000):
                #                 predicted_classes[i][j] = predicted_classes[i][j].argmax(axis=-1)
                #                 separated_classifier.append(predicted_classes[i][j].tolist())
                                
                #         matrix = confusion_matrix(true_classes, separated_classifier)
                #         plot_confusion_matrix(matrix, 0)
                #         plot_confusion_matrix(matrix, 1)
                
                # Calculates the matrix for average of the 3 classifiers of GoogLeNet
                final_classifier = []
                for i in range(2000):
                        line = [0, 0, 0, 0, 0]
                        sums = [0, 0, 0, 0, 0]
                        for j in range(3):
                                for number in range(5):
                                        sums[number] += predicted_classes[i][j][0][number]

                        for l in range(5):
                                line[l] = sums[l] / 3.0

                        print(line)
                        line = np.asarray(line)
                        line = line.argmax(axis=-1)
                        print(line)
                        final_classifier.append(line)
                
                matrix = confusion_matrix(true_classes, final_classifier)
                plot_confusion_matrix(matrix, 0)
                plot_confusion_matrix(matrix, 1)
        else:
                matrix = confusion_matrix(true_classes, predicted_classes)
                plot_confusion_matrix(matrix, 0)
                plot_confusion_matrix(matrix, 1)
        

### Plots the confusion matrix.
def plot_confusion_matrix(matrix, normalised):
        model_name = model_path[:-3]

        if (normalised == 0):
                #Without Normalisation:
                title = model_name + " - Confusion Matrix without Normalisation"
        else:
                #Normalised Matrix Block:
                title = model_name + " - Confusion Matrix Normalised"
                matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        fig, axis = plt.subplots()
        ### Change be
        image = axis.imshow(matrix, cmap = plt.cm.Greens)
        axis.figure.colorbar(image, ax=axis)

        # sets labels and ticks
        axis.set(
           yticks=np.arange(matrix.shape[0]),
           xticks=np.arange(matrix.shape[1]),
           yticklabels=labels, 
           xticklabels=labels,
           title=title,
           ylabel='True Class',
           xlabel='Predicted Class')
        
        # rotate the bottom labels 
        plt.setp(axis.get_xticklabels(), rotation=50, ha="right", rotation_mode="anchor")
        
        # make the text centered in the cell and change colour depending on the value below
        white_numbers = 200
        for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                        if matrix[row, column] > white_numbers:
                                color = "white"
                        else:
                                color = "black"

                        axis.text(column, row, format(matrix[row, column]),
                                ha = "center", 
                                va = "center",
                                color = color)

        plt.gcf().subplots_adjust(bottom=0.25)
        axis.set_ylim(len(matrix)-0.5, -0.5)
        fig.tight_layout()
        plt.show()

def main():
        #images = create_list_images(images_path)
        model = load_compiler_model()
        prepare_confusion_matrix(model)
        #predict(images, model)

main()