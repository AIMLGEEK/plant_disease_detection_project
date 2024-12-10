import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from plant_disease_detection_model import __version__ as _version
from plant_disease_detection_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

import tensorflow as tf

def dataset_scaling_reshaping(data_dir: str, subset_type: str):
    generator=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.0,
        preprocessing_function=None,
        validation_split=0.1
    ).flow_from_directory(data_dir,
                      config.batch_size,
                      target_size=(config.size, config.size),
                      subset=subset_type,
                      config.color_mode,
                      config.class_mode,
                      shuffle=True)
    
    return generator

def dataset_scaling_reshaping(data_dir: str):
    generator=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.0,
        preprocessing_function=None
    ).flow_from_directory(data_dir,
                      config.batch_size,
                      target_size=(config.size, config.size),
                      config.color_mode,
                      config.class_mode,
                      shuffle=True)
    
    return generator


##  Pre-Pipeline Preparation

# Extract year and month from the date column and create two another columns

def get_year_and_month(dataframe: pd.DataFrame, date_var: str):

    df = dataframe.copy()
    
    # convert 'dteday' column to Datetime datatype
    df[date_var] = pd.to_datetime(df[date_var], format='%Y-%m-%d')
    
    # Add new features 'yr' and 'mnth
    df['yr'] = df[date_var].dt.year
    df['mnth'] = df[date_var].dt.month_name()
    
    return df


def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame = get_year_and_month(dataframe = data_frame, date_var = config.model_config.date_var)
    
    # Drop unnecessary fields
    for field in config.model_config.unused_fields:
        if field in data_frame.columns:
            data_frame.drop(labels = field, axis=1, inplace=True)    

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame = dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous saved models. 
    This ensures that when the package is published, there is only one trained model that 
    can be called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
