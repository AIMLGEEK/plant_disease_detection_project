import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from plant_disease_detection_model.config.core import config

def get_class_names_from_directory(directory_path):
    # Load dataset from directory to extract class names
    dataset = image_dataset_from_directory(
        directory_path,
        label_mode='categorical',
        image_size=(config.app_config.size, config.app_config.size),
        batch_size=32
    )
    return dataset.class_names
