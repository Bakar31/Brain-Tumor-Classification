import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.optimizers import Adam
from data_generators import *

# building model
def model_EfficientNetB1(path, lr = 0.001, dr_rate = 0.15):
    model = EfficientNetB1(include_top=False, weights=path)
    model.trainable = False

    x = GlobalAveragePooling2D()(model.output)
    x = BatchNormalization()(x)
    x = Dropout(dr_rate)(x)
    dense_1 = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(dense_1)

    # Compile
    model = Model(model.inputs, outputs, name="EfficientNet")
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# model with flaire images
efficientB1_flaire = model_EfficientNetB1('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b1_tf24_imagenet_1000_notop.h5')
history_flair_2 = efficientB1_flaire.fit_generator(
        generator=train_flair,
        steps_per_epoch=len(train_flair),
        epochs=10,
        validation_data=test_flair,
        validation_steps=len(test_flair),
        callbacks=[callback],
        workers=2)

# model with t1w images
efficientB1_t1w = model_EfficientNetB1('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b1_tf24_imagenet_1000_notop.h5')
history_t1w_2 = efficientB1_t1w.fit_generator(
        generator=train_t1w,
        steps_per_epoch=len(train_t1w),
        epochs=10,
        validation_data=test_t1w,
        validation_steps=len(test_t1w),
        callbacks=[callback],
        workers=2)

# model with t1wce images
efficientB1_t1wce = model_EfficientNetB1('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b1_tf24_imagenet_1000_notop.h5')
history_t1wce_2 = efficientB1_t1wce.fit_generator(
        generator=train_t1wce,
        steps_per_epoch=len(train_t1wce),
        epochs=10,
        validation_data=test_t1wce,
        validation_steps=len(test_t1wce),
        callbacks=[callback],
        workers=2)

# model with t2w images
efficientB1_t2w = model_EfficientNetB1('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b1_tf24_imagenet_1000_notop.h5')
history_t2w_2 = efficientB1_t2w.fit_generator(
        generator=train_t2w,
        steps_per_epoch=len(train_t2w),
        epochs=10,
        validation_data=test_t2w,
        validation_steps=len(test_t2w),
        callbacks=[callback],
        workers=2)