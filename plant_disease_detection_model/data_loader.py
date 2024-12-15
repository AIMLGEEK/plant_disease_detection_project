from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape, batch_size=32, label_mode='categorical', augment=False, cache=True):
        """
        DataLoader transformer for loading datasets from directories.

        Args:
            input_shape (tuple): Shape of the input images (height, width).
            batch_size (int): Batch size for data loading.
            label_mode (str): Label mode ('categorical', 'binary', or None).
            augment (bool): Whether to apply data augmentation.
            cache (bool): Whether to cache the dataset.
        """
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.label_mode = label_mode
        self.augment = augment
        self.cache = cache

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform method to load datasets.

        Args:
            X (tuple): Tuple of (train_path, valid_path) directories.

        Returns:
            train_dataset, valid_dataset: Preprocessed training and validation datasets.
        """
        if isinstance(X, (tuple, list)) and len(X) == 2:
            train_path, valid_path = X
            logging.info(f"Loading datasets from: {train_path} and {valid_path}")
            train_dataset = self._load_dataset(train_path, is_training=True)
            valid_dataset = self._load_dataset(valid_path, is_training=False)
            return train_dataset, valid_dataset
        else:
            raise ValueError(
                f"Expected X to be a tuple of (train_path, valid_path), "
                f"but got {type(X)}. Ensure you pass the correct format."
            )

    def _load_dataset(self, path, is_training):
        """
        Load and preprocess the dataset.

        Args:
            path (Path or str): Path to the dataset directory.
            is_training (bool): Whether the dataset is for training.

        Returns:
            tf.data.Dataset: Loaded and preprocessed dataset.
        """
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            str(path),
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            label_mode=self.label_mode
        )

        if self.cache:
            dataset = dataset.cache()

        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)

        if self.augment and is_training:
            dataset = dataset.map(self._augment_data)

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def _augment_data(self, image, label):
        """
        Apply data augmentation.

        Args:
            image (tf.Tensor): Input image.
            label (tf.Tensor): Corresponding label.

        Returns:
            Augmented image and label.
        """
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2)
        ])
        return data_augmentation(image), label
