from DataFrameIterator import DCMDataFrameIterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 369
BATCH_SIZE = 512
CLASS_MODE = 'binary'
COLOR_MODE = 'rgb'
TARGET_SIZE = (224, 224)

def get_data_generators(train_df, test_df):
    train_augmentation_parameters = dict(
        zoom_range=0.2,
        fill_mode='nearest',
        height_shift_range= 0.1,
        width_shift_range=0.1,
        brightness_range = [0.8, 1.2]
    )

    test_augmentation_parameters = dict(
    #     rescale=1.0/255.0
    )

    train_consts = {
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'class_mode': CLASS_MODE,
        'color_mode': COLOR_MODE,
        'target_size': TARGET_SIZE,  
    }

    test_consts = {
        'batch_size': BATCH_SIZE,
        'class_mode': CLASS_MODE,
        'color_mode': COLOR_MODE,
        'target_size': TARGET_SIZE,
        'shuffle': False
    }

    train_augmenter = ImageDataGenerator(**train_augmentation_parameters)
    test_augmenter = ImageDataGenerator(**test_augmentation_parameters)

    train_generator = DCMDataFrameIterator(dataframe=train_df,
                                 x_col='file_paths',
                                 y_col='labels',
                                 image_data_generator=train_augmenter,
                                 **train_consts)

    test_generator = DCMDataFrameIterator(dataframe=test_df,
                                 x_col='file_paths',
                                 y_col='labels',
                                 image_data_generator=test_augmenter,
                                 **test_consts)
    
    return train_generator, test_generator