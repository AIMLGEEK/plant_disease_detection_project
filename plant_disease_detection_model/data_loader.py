import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageDataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, img_size, batch_size):
        self.img_size = img_size
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        datagen = ImageDataGenerator(rescale=1/255.0)
        generator = datagen.flow_from_directory(
            X,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        return generator
