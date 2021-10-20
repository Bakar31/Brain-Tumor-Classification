import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.optimizers import Adam
from data_generators import *

# building model
def model_EfficientNetB1(path, lr = 0.001, dr_rate = 0.15):
    model = EfficientNetB1(include_top=False, weights=path)
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D()(model.output)
    x = BatchNormalization()(x)
    x = Dropout(dr_rate)(x)
    dense_1 = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(dense_1)

    # Compile
    model = Model(model.inputs, outputs, name="EfficientNet")
    optimizer = Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model