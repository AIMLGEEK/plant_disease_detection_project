import pickle

from keras.src.utils import image_dataset_from_directory
from sklearn.pipeline import Pipeline
from plant_disease_detection_model.config.core import config, DATASET_DIR
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import json


class DataLoader(BaseEstimator, TransformerMixin):
    """Enhanced data loader with performance optimizations."""

    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            batch_size: int = 32,
            label_mode: str = 'categorical',
            augment: bool = False,
            cache: bool = True
    ):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.label_mode = label_mode
        self.augment = augment
        self.cache = cache
        self.logger = logging.getLogger(self.__class__.__name__)

        self._augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2)
        ])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, (tuple, list)) or len(X) != 2:
            raise ValueError("Expected tuple of (train_path, valid_path)")

        train_path, valid_path = X
        self.logger.info(f"Loading datasets from: {train_path} and {valid_path}")

        train_dataset = self._create_dataset(train_path, is_training=True)
        valid_dataset = self._create_dataset(valid_path, is_training=False)

        return train_dataset, valid_dataset

    def _create_dataset(self, path: Path, is_training: bool) -> tf.data.Dataset:
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(path),
            image_size=self.input_shape[:2],
            batch_size=self.batch_size,
            label_mode=self.label_mode,
            shuffle=is_training
        )

        # Performance optimizations
        dataset = dataset.map(
            self._preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if is_training and self.augment:
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        if self.cache:
            dataset = dataset.cache()

        if is_training:
            dataset = dataset.shuffle(1000)

        return dataset.prefetch(tf.data.AUTOTUNE)

    def _preprocess(self, images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    def _augment(self, images, labels):
        return self._augmentation(images, training=True), labels


class EnsembleModel(BaseEstimator, TransformerMixin):
    """Enhanced ensemble model with improved architecture."""

    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            num_classes: int,
            learning_rate: float = 0.001,
            mobilenet_weight: float = 0.5,
            resnet_weight: float = 0.5
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mobilenet_weight = mobilenet_weight
        self.resnet_weight = resnet_weight
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.history = None

    def _build_mobilenet(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)

    def _build_resnet(self):
        base_model = tf.keras.applications.ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )

        # Unfreeze last few layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False

        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)

    def fit(self, X, y=None):
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("Expected tuple of (train_dataset, valid_dataset)")

        train_dataset, valid_dataset = X
        self.logger.info("Building ensemble model")

        # Build models
        mobilenet = self._build_mobilenet()
        resnet = self._build_resnet()

        # Create ensemble
        inputs = tf.keras.Input(shape=self.input_shape)
        mobilenet_output = mobilenet(inputs)
        resnet_output = resnet(inputs)

        ensemble_output = tf.keras.layers.Average()(
            [self.mobilenet_weight * mobilenet_output,
             self.resnet_weight * resnet_output]
        )

        self.model = tf.keras.Model(inputs, ensemble_output)

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Train model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/saved_model/best_model.keras',
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=0.000001
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='models/logs',
                histogram_freq=1
            )
        ]

        history = self.model.fit(
            train_dataset,
            epochs=config.self_model_config.epochs,
            validation_data=valid_dataset,
            callbacks=callbacks
        )

        # Store the history
        self.history = history.history
        return self

    def transform(self, X):
        return self.model.predict(X)

    def save(self, filepath: Path):
        self.model.save(filepath / 'final_model')

    @classmethod
    def load(cls, filepath: Path, input_shape, num_classes):
        instance = cls(input_shape, num_classes)
        instance.model = tf.keras.models.load_model(filepath / 'final_model')
        return instance

    def get_training_history(self):
        return self.history

class ClassNameSaver(BaseEstimator, TransformerMixin):
    """Saves class names from the dataset."""

    def __init__(self, dataset_directory, save_path: Path):
        self.dataset_directory = dataset_directory
        self.save_path = save_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X, y=None):
        dataset, _ = X

        # Load dataset from directory to extract class names
        dataset = image_dataset_from_directory(
            self.dataset_directory,
            label_mode='categorical',
            image_size=(config.app_config.size, config.app_config.size),
            batch_size=32
        )

        class_names = dataset.class_names

        self.logger.info(f"Saving {len(class_names)} class names")
        with open(self.save_path / 'class_names.json', 'w') as f:
            json.dump(class_names, f)

        return self

    def transform(self, X):
        return X


def create_pipeline(config: dict) -> Pipeline:
    """Creates the training pipeline."""
    return Pipeline([
        ('data_loader', DataLoader(
            input_shape=config['input_shape'],
            batch_size=config['batch_size'],
            augment=config['augment']
        )),
        ('class_names', ClassNameSaver(
            dataset_directory = config['dataset_directory'],
            save_path=Path(config['model_dir'])
        )),
        ('model', EnsembleModel(
            input_shape=config['input_shape'],
            num_classes=config['num_classes'],
            learning_rate=config['learning_rate']
        ))
    ])

def train_model():
    img_size = (config.app_config.size, config.app_config.size)
    batch_size = config.self_model_config.batch_size
    num_classes = config.self_model_config.num_classes
    learning_rate = config.self_model_config.learning_rate
    dataset_directory = Path(DATASET_DIR / config.app_config.training_data_folder)
    input_shape = img_size + (3,)
    train_config = {
        'input_shape': input_shape,
        'batch_size': batch_size,
        'num_classes': num_classes,
        'learning_rate': learning_rate,
        'augment': True,
        'model_dir': Path('models/saved_model'),
        'dataset_directory': dataset_directory
    }

    train_dir = Path(DATASET_DIR / 'augmented_data/train')
    valid_dir = Path(DATASET_DIR / 'augmented_data/valid')

    pipeline = create_pipeline(train_config)
    pipeline.fit((train_dir, valid_dir))

    # Save the pipeline
    with open('models/saved_model/model_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

if __name__ == "__main__":
    train_model()
