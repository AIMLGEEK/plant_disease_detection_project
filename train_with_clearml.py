import shutil

from clearml import Task, Dataset, OutputModel, StorageManager
from pathlib import Path
import json
import os
import pickle
import logging
import tensorflow as tf
from pipeline import create_pipeline
from config.core import config, DATASET_DIR
from model_version import ModelVersioning

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_clearml():
    """Initialize ClearML configuration"""
    Task.set_credentials(
        api_host="https://api.clear.ml",
        web_host="https://app.clear.ml",
        files_host="https://files.clear.ml",
        key="835FC4669AGN89YFOB250Q0YPVQGPT",
        secret="r_2ii52kYIlU-LGe7wrj4OPQ4nCDiAilchZcAPsDft11mWTfNU-nMMaLUexuyeLhY74"
    )

    # Set default upload location
    StorageManager.set_cache_file_limit(5, cache_context=None)

def verify_dataset(dataset_path: Path):
    """
    Verify dataset exists and contains image files
    """
    logger.info(f"Verifying dataset at {dataset_path}")

    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Count image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = sum(1 for f in dataset_path.rglob('*')
                      if f.is_file() and f.suffix.lower() in image_extensions)

    logger.info(f"Found {image_files} image files in dataset")

    if image_files == 0:
        raise ValueError("No image files found in dataset")

    # Count classes (directories)
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(classes)} classes: {[d.name for d in classes]}")

    return True


def save_model_safely(model, save_path: Path, backup_dir: Path = None):
    """
    Safely save the model with Keras 3 compatibility and enhanced error handling.

    Args:
        model: TensorFlow model to save
        save_path: Primary path to save the model
        backup_dir: Optional backup directory path

    Returns:
        bool: True if save successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    # Create temporary directory for saving
    temp_dir = Path('temp_model_save')
    temp_dir.mkdir(exist_ok=True)
    temp_save_path = temp_dir / 'temp_model.keras'

    try:
        # First try saving to temporary location
        logger.info(f"Attempting to save model to temporary location: {temp_save_path}")
        tf.keras.models.save_model(
            model,
            str(temp_save_path),
            include_optimizer=False
        )

        # Ensure target directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # If backup_dir provided, create backup
        if backup_dir and save_path.exists():
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"backup_{save_path.name}"
            shutil.copy2(save_path, backup_path)
            logger.info(f"Created backup at: {backup_path}")

        # Move temporary save to final location
        if save_path.exists():
            save_path.unlink()  # Remove existing file if present
        shutil.move(str(temp_save_path), str(save_path))
        logger.info(f"Model saved successfully to {save_path}")

        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    except PermissionError as e:
        logger.error(f"Permission denied while saving model: {str(e)}")
        logger.info("Attempting to adjust permissions...")
        try:
            # Try to adjust permissions of parent directory
            os.chmod(save_path.parent, 0o777)
            # Retry save
            tf.keras.models.save_model(
                model,
                str(save_path),
                include_optimizer=False
            )
            logger.info("Successfully saved model after adjusting permissions")
            return True
        except Exception as inner_e:
            logger.error(f"Failed to save model even after adjusting permissions: {str(inner_e)}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error while saving model: {str(e)}")
        return False

    finally:
        # Always clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_model_directories():
    """
    Create necessary directories for model saving with appropriate permissions.

    Returns:
        Path: Path to the model directory
    """
    model_dir = Path('models/saved_model')
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.chmod(model_dir, 0o777)
    except Exception as e:
        logging.warning(f"Could not set directory permissions: {str(e)}")

    return model_dir

def train_model():
    # Initialize ClearML
    init_clearml()
    logger.info("Initialized ClearML")
    model_dir = create_model_directories()
    # Create ClearML task
    task = Task.init(
        project_name="plant_disease_detection_project",
        task_name="MobilenetV2_training",
        auto_connect_frameworks={'tensorflow': False}
    )
    logger.info("Created ClearML task")

    # Initialize model versioning
    versioning = ModelVersioning()

    # Setup training configuration
    img_size = (config.app_config.size, config.app_config.size)
    input_shape = img_size + (3,)

    # Log configuration
    task.connect_configuration(
        name="model_config",
        configuration=config.dict()
    )

    try:
        # Create ClearML dataset
        dataset = Dataset.get(
            dataset_project="plant_disease_detection_project",
            dataset_name="plant_disease_detection_dataset"
        )

        # Get local copy of dataset
        local_dataset_path = Path(dataset.get_local_copy())
        logger.info(f"Retrieved local dataset copy at: {local_dataset_path}")

        verify_dataset(dataset_path=local_dataset_path)

        # Create and train pipeline
        logger.info("Creating and training pipeline...")

        train_config = {
            'input_shape': input_shape,
            'batch_size': config.self_model_config.batch_size,
            'learning_rate': config.self_model_config.learning_rate,
            'augment': True,
            'model_dir': Path('models/saved_model'),
            'dataset_directory': local_dataset_path
        }
        pipeline = create_pipeline(train_config)
        history = pipeline.fit((local_dataset_path, local_dataset_path))

        # Get training history from the model step
        history = pipeline.named_steps['model'].get_training_history()

        training_metrics = {
            'final_loss': float(history['loss'][-1]),
            'final_accuracy': float(history['accuracy'][-1]),
            'val_loss': float(history['val_loss'][-1]),
            'val_accuracy': float(history['val_accuracy'][-1])
        }
        version = versioning.update_version(training_metrics)
        logger.info(f"Updated model version to {version}")

        # Save model and artifacts locally first
        model_dir = Path('models/saved_model')
        model_save_path = model_dir / f'model_v{version}.keras'
        backup_dir = Path('models/backups')

        # Get the trained model from the pipeline
        model = pipeline.named_steps['model'].model

        if not save_model_safely(model, model_save_path, backup_dir):
            logger.error("Model saving failed")
            raise Exception("Failed to save model")

        # Save pipeline separately
        pipeline_save_path = model_dir / f'pipeline_v{version}.pkl'
        with open(pipeline_save_path, 'wb') as f:
            pickle.dump(pipeline, f)

        metadata = {
            'version': version,
            'config': {k: str(v) if isinstance(v, Path) else v
                       for k, v in train_config.items()},
            'metrics': training_metrics
        }

        metadata_path = model_dir / f'metadata_v{version}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Log artifacts to ClearML
        logger.info("Uploading artifacts to ClearML...")

        # Upload model files
        task.upload_artifact(
            name=f"model_v{version}",
            artifact_object=model_save_path,
        )

        # Upload pipeline
        task.upload_artifact(
            name=f"pipeline_v{version}",
            artifact_object=pipeline_save_path,
        )

        # Upload metadata
        task.upload_artifact(
            name="metadata",
            artifact_object=metadata
        )

        # Upload version info
        task.upload_artifact(
            name="version_info",
            artifact_object=versioning.version_info
        )

        logger.info(f"Model training completed successfully with version {version}")
        return task

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    train_model()