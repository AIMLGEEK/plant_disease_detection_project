import pickle
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from config.core import config, DATASET_DIR
from pipeline import DataLoader, EnsembleModel, ClassNameSaver  # Add this import
import logging

logger = logging.getLogger(__name__)


def load_pipeline(model_path: Path):
    """Load the trained model pipeline."""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model pipeline: {e}")
        raise


def load_class_names(class_names_path: Path) -> list:
    """Load class names from JSON file."""
    try:
        with open(class_names_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        raise


def preprocess_image(img_path: Path, target_size: tuple) -> np.ndarray:
    """
    Preprocess the image to be compatible with the model input.

    Args:
        img_path: Path to the image file
        target_size: Tuple of (height, width) for resizing

    Returns:
        Preprocessed image array
    """
    try:
        img = tf.keras.utils.load_img(str(img_path), target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = tf.cast(img_array, tf.float32) / 255.0  # Normalize to [0,1]
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def make_prediction(model_path: Path, img_path: Path) -> dict:
    """
    Make a prediction using the trained model pipeline.

    Args:
        model_path: Path to the saved model pipeline
        img_path: Path to the image file

    Returns:
        Dictionary containing prediction results and confidence scores
    """
    try:
        # Load model pipeline and class names
        pipeline = load_pipeline(model_path / 'model_pipeline.pkl')
        class_names = load_class_names(model_path / 'class_names.json')

        # Get model from pipeline and preprocess image
        model = pipeline.named_steps['model']
        input_shape = model.input_shape
        img_array = preprocess_image(img_path, input_shape[:2])

        # Make prediction
        predictions = model.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence_score = float(predictions[0][predicted_class_idx])

        # Format results
        result = {
            'predicted_class': class_names[predicted_class_idx],
            'confidence': confidence_score,
            'all_probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(class_names, predictions[0])
            }
        }

        logger.info(f"Prediction made for {img_path}: {result['predicted_class']} "
                    f"with confidence {result['confidence']:.2%}")

        return result

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    # Set up paths
    model_path = Path('models/saved_model')
    img_path = DATASET_DIR/"augmented_data/train/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417_90deg.JPG"  # Update with your image path

    try:
        # Make prediction
        result = make_prediction(model_path, img_path)

        # Print results
        print(f"\nPrediction Results:")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
        )[:3]:  # Show top 3 predictions
            print(f"{class_name}: {prob:.2%}")

    except Exception as e:
        print(f"Error during prediction: {e}")