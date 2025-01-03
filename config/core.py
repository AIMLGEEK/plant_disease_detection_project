# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

# Project Directories
#PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent
CONFIG_FILE_PATH = ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = ROOT / "datasets"
TRAINED_MODEL_DIR = ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    training_data_folder: str
    size: int


class ModelConfig(BaseModel):
    test_size:float
    random_state: int
    n_estimators: int
    max_depth: int
    batch_size: int
    epochs: int
    learning_rate: float
    mobilenet_weight: float
    resnet_weight: float


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    self_model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    print(CONFIG_FILE_PATH)
    print(ROOT)
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    print(CONFIG_FILE_PATH)
    print(ROOT)
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        self_model_config = ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()