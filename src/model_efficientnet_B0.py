import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.optimizers import Adam
from data_generators import *

def model_EfficientNetB0(path, lr = 0.001, dr_rate = 0.15):
    model = EfficientNetB0(include_top=False, weights=path)
    model.trainable = False

    x = GlobalAveragePooling2D()(model.output)
    x = BatchNormalization()(x)
    x = Dropout(dr_rate)(x)
    dense_1 = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(dense_1)

    model = Model(model.inputs, outputs, name="EfficientNet")
    optimizer = Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"]
    )
    return model
# callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# model with flaire images
efficientB0_flaire = model_EfficientNetB0('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b0_tf24_imagenet_1000_notop.h5')
history_flair_1 = efficientB0_flaire.fit_generator(
        generator=train_flair,
        steps_per_epoch=len(train_flair),
        epochs=10,
        validation_data=test_flair,
        validation_steps=len(test_flair),
        callbacks=[callback],
        workers=2)

# model with t1w images
efficientB0_t1w = model_EfficientNetB0('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b0_tf24_imagenet_1000_notop.h5')
history_t1w_1 = efficientB0_t1w.fit_generator(
        generator=train_t1w,
        steps_per_epoch=len(train_t1w),
        epochs=10,
        validation_data=test_t1w,
        validation_steps=len(test_t1w),
        callbacks=[callback],
        workers=2)

# model with t1wce images
efficientB0_t1wce = model_EfficientNetB0('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b0_tf24_imagenet_1000_notop.h5')
history_t1wce_1 = efficientB0_t1wce.fit_generator(
        generator=train_t1wce,
        steps_per_epoch=len(train_t1wce),
        epochs=10,
        validation_data=test_t1wce,
        validation_steps=len(test_t1wce),
        callbacks=[callback],
        workers=2)

# model with t2w images
efficientB0_t2w = model_EfficientNetB0('../input/efficentnet-b0b5-tensorflow-24-notop/efficientnet-b0_tf24_imagenet_1000_notop.h5')
history_t2w_1 = efficientB0_t2w.fit_generator(
        generator=train_t2w,
        steps_per_epoch=len(train_t2w),
        epochs=10,
        validation_data=test_t2w,
        validation_steps=len(test_t2w),
        callbacks=[callback],
        workers=2)