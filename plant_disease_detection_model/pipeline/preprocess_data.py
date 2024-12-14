import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from plant_disease_detection_model.config.core import config, DATASET_DIR

def split_data(src_dir, train_dir, valid_dir, test_dir, valid_size=0.1, test_size=0.1):
    categories = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for category in categories:
        category_path = os.path.join(src_dir, category)
        images = os.listdir(category_path)
        np.random.shuffle(images)

        train_and_valid, test = train_test_split(images, test_size=test_size)
        train, valid = train_test_split(train_and_valid, test_size=valid_size / (1 - test_size))

        for image in train:
            shutil.copy(os.path.join(category_path, image), os.path.join(train_dir, category, image))
        for image in valid:
            shutil.copy(os.path.join(category_path, image), os.path.join(valid_dir, category, image))
        for image in test:
            shutil.copy(os.path.join(category_path, image), os.path.join(test_dir, category, image))


def preprocess_data():

    src_dir = DATASET_DIR
    train_dir = 'data/train'
    valid_dir = 'data/valid'
    test_dir = 'data/test'

    split_data(src_dir, train_dir, valid_dir, test_dir)

    # Prepare data generators
    size = config.self_model_config.num_classes
    batch_size = config.self_model_config.batch_size

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0,
        validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=size,
        batch_size=batch_size,
        subset="training",
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = train_datagen.flow_from_directory(
        valid_dir,
        target_size=size,
        batch_size=batch_size,
        subset="validation",
        class_mode='categorical',
        shuffle=False
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, valid_generator, test_generator


if __name__ == "__main__":
    preprocess_data()