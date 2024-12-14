import joblib
from sklearn.pipeline import Pipeline
from data_loader import ImageDataLoader
from model import MobileNetV2Wrapper
from plant_disease_detection_model.config.core import config, DATASET_DIR

def train_model():

    img_size = (config.app_config.size, config.app_config.size)
    batch_size = config.self_model_config.batch_size
    num_classes = config.self_model_config.num_classes
    input_shape = img_size + (3,)

    train_dir = DATASET_DIR / 'augmented_data/train'
    valid_dir = DATASET_DIR / 'augmented_data/valid'

    data_loader = ImageDataLoader(img_size, batch_size)
    mobilenet_v2 = MobileNetV2Wrapper(input_shape, num_classes)

    pipeline = Pipeline([
        ('data_loader', data_loader),
        ('mobilenet_v2', mobilenet_v2)
    ])

    pipeline.fit(train_dir, valid_dir)

    # Save the trained model
    joblib.dump(pipeline, 'models/saved_model/model_pipeline.pkl')

if __name__ == "__main__":
    train_model()
