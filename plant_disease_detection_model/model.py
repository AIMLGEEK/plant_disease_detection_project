import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from tensorflow.data import Dataset
from plant_disease_detection_model.config.core import config, DATASET_DIR
from plant_disease_detection_model.utils.classname_extractor import get_class_names_from_directory

class EnsembleModelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = config.self_model_config.learning_rate
        self.mobilenet_weight = config.self_model_config.mobilenet_weight
        self.resnet_weight = config.self_model_config.resnet_weight
        self.model = self._build_ensemble_model()
        self.train_dir = DATASET_DIR / config.app_config.training_data_folder

    def _build_mobilenet_model(self):
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape)
        base_model.trainable = False

        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs, outputs)

    def _build_resnet_model(self):
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
        inputs = Input(shape=self.input_shape)

        # Unfreeze last 10 layers for fine-tuning
        for layer in base_model.layers[:-10]:
            layer.trainable = False

        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs, outputs)

    def _build_ensemble_model(self):
        # Create individual models
        mobilenet = self._build_mobilenet_model()
        resnet = self._build_resnet_model()

        # Combine their outputs
        combined_input = Input(shape=self.input_shape)
        mobilenet_output = mobilenet(combined_input)
        resnet_output = resnet(combined_input)

        # Weighted average of outputs
        combined_output = Add()([
            self.mobilenet_weight * mobilenet_output,
            self.resnet_weight * resnet_output
        ])

        # Create ensemble model
        model = Model(inputs=combined_input, outputs=combined_output)

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def fit(self, X, y=None):
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("Expected X to be a tuple of (train_dataset, valid_dataset).")

        train_dataset, valid_dataset = X

        if not isinstance(train_dataset, Dataset) or not isinstance(valid_dataset, Dataset):
            raise ValueError("train_dataset and valid_dataset must be TensorFlow Dataset objects.")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(Path('models/saved_model/best_model.keras')),
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

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath, input_shape, num_classes, learning_rate):
        loaded_model = tf.keras.models.load_model(filepath)
        instance = cls(input_shape, num_classes, learning_rate)
        instance.model = loaded_model
        return instance
