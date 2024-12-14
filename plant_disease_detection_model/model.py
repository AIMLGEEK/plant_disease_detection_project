import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from plant_disease_detection_model.config.core import config, DATASET_DIR
from plant_disease_detection_model.utils.classname_extractor import get_class_names_from_directory

class MobileNetV2Wrapper(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.train_dir = DATASET_DIR / config.app_config.training_data_folder

    def _build_model(self):
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape)
        base_model.trainable = False

        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'precision', 'recall'])

        return model

    def fit(self, X, y=None):
        """
        Expects X as a tuple (train_dataset, valid_dataset) from DataLoader.
        """
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("Expected X to be a tuple of (train_dataset, valid_dataset).")

        train_dataset, valid_dataset = X

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                str(Path('models/saved_model/best_model.keras')),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001)
        ]

        self.model.fit(
            train_dataset,
            epochs=config.self_model_config.epochs,
            validation_data=valid_dataset,
            callbacks=callbacks
        )
        return self

    def predict(self, X):
        # Predict probabilities
        predictions = self.model.predict(X)

        # Get predicted class IDs
        predicted_class_ids = predictions.argmax(axis=-1)

        # Dynamically load class names from the training directory
        class_names = get_class_names_from_directory(self.train_dir)

        # Map class IDs to class names
        predicted_class_names = [class_names[idx] for idx in predicted_class_ids]

        return predicted_class_names
