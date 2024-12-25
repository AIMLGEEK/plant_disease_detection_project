import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config.core import config

def create_dir(dir_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def split_data(src_dir, train_dir, valid_dir, valid_size=0.1):
    """Split data into training and validation sets."""
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
    """Copy file from source to destination."""
    try:
        shutil.copy(src, dst)
    except PermissionError:
        print(f"Permission denied: {src}")
    except Exception as e:
        print(f"Error copying file {src} to {dst}: {e}")

def preprocess_data(src_dir, train_dir, valid_dir, valid_size=0.1):
    """
    Main preprocessing function to split and prepare datasets.

    Args:
        src_dir (Path): Path to the source directory with raw data.
        train_dir (Path): Path to save the training data.
        valid_dir (Path): Path to save the validation data.
        valid_size (float): Proportion of validation data.
    """
    # Ensure output directories exist
    create_dir(train_dir)
    create_dir(valid_dir)

    # Split data
    split_data(src_dir, train_dir, valid_dir, valid_size)

    # Prepare data generators
    batch_size = config.self_model_config.batch_size  # You can adjust this based on your needs
    image_size = (config.app_config.size, config.app_config.size)  # Replace with your desired image size

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0  # Normalizing pixel values to [0, 1]
    )

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0  # Normalizing pixel values to [0, 1]
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, valid_generator

if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Preprocess raw dataset.")
    parser.add_argument("--src_dir", type=str, required=True, help="Path to the raw dataset directory.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to save the training dataset.")
    parser.add_argument("--valid_dir", type=str, required=True, help="Path to save the validation dataset.")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Validation split size (default: 0.1).")

    args = parser.parse_args()

    train_gen, valid_gen = preprocess_data(
        src_dir=args.src_dir,
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        valid_size=args.valid_size
    )

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {valid_gen.samples}")
