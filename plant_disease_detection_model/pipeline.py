import joblib
from sklearn.pipeline import Pipeline
from data_loader import DataLoader
from model import EnsembleModelWrapper
from plant_disease_detection_model.config.core import config, DATASET_DIR
from pathlib import Path

def train_model():

    img_size = (config.app_config.size, config.app_config.size)
    batch_size = config.self_model_config.batch_size
    num_classes = config.self_model_config.num_classes
    input_shape = img_size + (3,)

    train_dir = Path(DATASET_DIR / 'augmented_data/train')
    valid_dir = Path(DATASET_DIR / 'augmented_data/valid')

    data_loader = DataLoader(img_size, batch_size)
    ensemble_model = EnsembleModelWrapper(input_shape, num_classes)

    pipeline = Pipeline([
        ('data_loader', data_loader),
        ('ensemble_model', ensemble_model)
    ])

    pipeline.fit((train_dir, valid_dir))

    # Save the trained model
    joblib.dump(pipeline, 'models/saved_model/model_pipeline.pkl')

if __name__ == "__main__":
    train_model()
