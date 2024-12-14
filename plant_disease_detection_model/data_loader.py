from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import tensorflow as tf

class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape, batch_size=32, label_mode='categorical'):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.label_mode = label_mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, (tuple, list)) and len(X) == 2:
            train_path, valid_path = X
            train_dataset = self._load_dataset(train_path)
            valid_dataset = self._load_dataset(valid_path)
            return train_dataset, valid_dataset
        elif isinstance(X, (str, Path)):
            raise ValueError(
                f"Expected X to be a tuple of (train_path, valid_path), "
                f"but got a single path {X} of type {type(X)}. Ensure you pass a tuple."
            )
        else:
            raise ValueError(f"Unexpected input type for X: {type(X)}. Expected tuple, str, or Path.")

    def _load_dataset(self, path):
        return tf.keras.preprocessing.image_dataset_from_directory(
            str(path),
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            label_mode=self.label_mode
        )