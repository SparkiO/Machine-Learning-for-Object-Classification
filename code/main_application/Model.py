import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image
import numpy as np

class BaseModel:
    def __init__(self, path):
        self.model = load_model(path)

    def normalise(self, data):
        return data

    def _predict(self, data):
        return self.model.predict(data)[0]

    def predict(self, path):
        img = image.load_img(path, target_size=(224,224))
        data = np.asarray(img, dtype="float32")
        data = np.expand_dims(data, axis=0)
        data = self.normalise(data)
        return self._predict(data)

class ZFNet(BaseModel):
    def normalise(self, data):
        return data / 255

class GoogLeNet(BaseModel):
    def _predict(self, data):
        google_predict = self.model.predict(data)
        average_vec = []
        for i in range (5):
            sum = 0.0
            for j in range (3):
                    sum += google_predict[j][0][i]
            average = sum / 3
            average_vec.append(average*100)
        return average_vec
