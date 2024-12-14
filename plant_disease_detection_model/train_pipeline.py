import tensorflow as tf
import yaml
from model import build_model
from pipeline.preprocess_data import preprocess_data


def train_model(config_path='src/config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    input_shape = config.target_size
    num_classes = config['data']['num_classes']
    epochs = config['training']['epochs']
    batch_size = config['data']['batch_size']

    train_generator, valid_generator = preprocess_data(config_path)

    model = build_model(input_shape, num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('models/saved_model/best_model.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001)
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=callbacks
    )


if __name__ == "__main__":
    train_model()
