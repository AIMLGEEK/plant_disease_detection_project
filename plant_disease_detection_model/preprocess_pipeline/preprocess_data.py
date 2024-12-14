import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from plant_disease_detection_model.config.core import config, DATASET_DIR

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def split_data(src_dir, train_dir, valid_dir, valid_size=0.1):
    categories = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for category in categories:
        category_path = os.path.join(src_dir, category)

        images = [os.path.join(category, img) for img in os.listdir(category_path) if
                  os.path.isfile(os.path.join(category_path, img))]
        np.random.shuffle(images)

        train, valid = train_test_split(images, test_size=valid_size)

        create_dir(os.path.join(train_dir, category))
        create_dir(os.path.join(valid_dir, category))

        for image in train:
            copy_file(os.path.join(src_dir, image), os.path.join(train_dir, image))
        for image in valid:
            copy_file(os.path.join(src_dir, image), os.path.join(valid_dir, image))

def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except PermissionError:
        print(f"Permission denied: {src}")
    except Exception as e:
        print(f"Error copying file {src} to {dst}: {e}")

def preprocess_data():

    src_dir = DATASET_DIR / config.app_config.training_data_folder
    train_dir = DATASET_DIR / 'augmented_data/train'
    valid_dir = DATASET_DIR / 'augmented_data/valid'

    create_dir(train_dir)
    create_dir(valid_dir)

    split_data(src_dir, train_dir, valid_dir)

    # Prepare data generators
    size = config.self_model_config.num_classes
    batch_size = config.self_model_config.batch_size

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0,
        validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(config.app_config.size, config.app_config.size),
        batch_size=batch_size,
        subset="training",
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = train_datagen.flow_from_directory(
        valid_dir,
        target_size=(config.app_config.size, config.app_config.size),
        batch_size=batch_size,
        subset="validation",
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, valid_generator


if __name__ == "__main__":
    preprocess_data()