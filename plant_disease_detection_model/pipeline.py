import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from plant_disease_detection_model.config.core import config
from plant_disease_detection_model.processing.features import WeekdayImputer, WeathersitImputer
from plant_disease_detection_model.processing.features import Mapper
from plant_disease_detection_model.processing.features import OutlierHandler, WeekdayOneHotEncoder

plant_disease_detection_pipe = Pipeline([

    ######### Imputation ###########
    ('weekday_imputation', WeekdayImputer(variable = config.model_config.weekday_var, 
                                          date_var= config.model_config.date_var)),
    ('weathersit_imputation', WeathersitImputer(variable = config.model_config.weathersit_var)),
    
    ######### Mapper ###########
    ('map_yr', Mapper(variable = config.model_config.yr_var, mappings = config.model_config.yr_mappings)),
    
    ######## Handle outliers ########
    ('handle_outliers_temp', OutlierHandler(variable = config.model_config.temp_var)),

    ######## One-hot encoding ########
    ('encode_weekday', WeekdayOneHotEncoder(variable = config.model_config.weekday_var)),

    # Scale features
    ('scaler', StandardScaler()),
    
    # Dataset pre processing
    ('train_generator', dataset_scaling_reshaping(variable= config.train_dir, "training")),
    ('validation_generator', dataset_scaling_reshaping(variable= config.validation_dir, "validation")),
    ('test_generator', dataset_scaling_reshaping(variable= config.test_dir)),
    
    # MobileNetv2
    ('model_mobilenetv2', model.fit(train_generator,
                    config.epochs,
                    batch_size=None,
                    validation_data = valid_generator,
                    callbacks = callbacks
                    ))
    
    ])
