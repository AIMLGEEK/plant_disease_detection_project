import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
from plant_disease_detection_model.utils.classname_extractor import get_class_names_from_directory
from plant_disease_detection_model.config.core import config, DATASET_DIR

def load_pipeline(model_path):
    """Load the trained model pipeline."""
    return joblib.load(model_path)


def preprocess_image(img_path, target_size):
    """Preprocess the image to be compatible with the model input."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescaling
    return img_array


def make_prediction(model_path, img_path):
    """Make a prediction using the trained model pipeline."""

    # Load model pipeline
    pipeline = load_pipeline(model_path)

    # Preprocess image
    input_shape = pipeline.named_steps['mobilenet_v2'].input_shape
    img_array = preprocess_image(img_path, input_shape[:2])

    # Predict using the model
    prediction = pipeline.named_steps['mobilenet_v2'].model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    # Dynamically load class names from the training directory
    class_names = get_class_names_from_directory(DATASET_DIR / config.app_config.training_data_folder)

    # Map class IDs to class names
    predicted_class_names = [class_names[idx] for idx in predicted_class]

    return predicted_class_names


if __name__ == "__main__":
    model_path = 'models/saved_model/model_pipeline.pkl'
    img_path = 'D:\AppleCedarRust1.JPG'

    prediction = make_prediction(model_path, img_path)
    print(f"Predicted class: {prediction}")
