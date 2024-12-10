import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from plant_disease_detection_model.config.core import config
from plant_disease_detection_model.pipeline import plant_disease_detection_pipe
from plant_disease_detection_model.processing.data_manager import load_dataset, save_pipeline

def create_MobileNetV2_model()

    input_shape = (config.size, config.size, 3)

    inputs = tf.keras.Input(shape=input_shape)

    model_name='MobileNetV2'
    # Loading the base model without classification head. Using imagenet weights.
    base_model = models[model_name](include_top=False, weights='imagenet', input_tensor=inputs)
    # Freeze the base model layers
    base_model.trainable = False

    #Adding additonal layers
    #Flattening and global average pooling, dense layers are a must
    #Adding Batch normalization reduces covariate shift - https://arxiv.org/abs/1502.03167
    # Dropout avoids overfitting - https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    x = tf.keras.layers.Flatten()(base_model.output)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(38, activation='softmax')(x)

    # Create model
    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)

    # Compiling the model
    optimizer = tf.keras.optimizers.Adam(config.learning_rate=0.001, config.beta_1=0.9, config.beta_2=0.999)

    model.compile(loss="categorical_crossentropy", optimizer= optimizer, metrics=["accuracy",'precision','recall'])

    # Display summary
    model.summary()
    
    return model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

def early_stopping_callback():
    return EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

def model_checkpoint_callback():
    return ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

def model_ReduceLROnPlateau_callback():
    return ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001)

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    early_stopping = early_stopping_callback()
    model_checkpoint = model_checkpoint_callback()
    model_ReduceLROnPlateau = model_ReduceLROnPlateau_callback()
    
    callbacks=[early_stopping,model_checkpoint,model_ReduceLROnPlateau]
    
    model = create_MobileNetV2_model()
    
    history = model.fit(train_generator,
                    config.epochs,
                    batch_size=None,
                    validation_data = valid_generator,
                    callbacks = callbacks
                    )

    # divide train and test

    # Pipeline fitting
    plant_disease_detection_pipe.fit(model)

    # Calculate the score/error
    #print("R2 score:", r2_score(y_test, y_pred).round(2))
    #print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist = plant_disease_detection_pipe)
    
if __name__ == "__main__":
    run_training()