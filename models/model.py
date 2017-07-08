from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, AveragePooling2D
from image_loading.image_loader import eager_load_data
from keras.regularizers import l2
from image_loading.utils import save_array, load_array

from keras.optimizers import Adam
import numpy as np
from keras.applications import VGG16
import pdb
from keras.models import Model
def simple_cnn():
    inps = Input(shape=(256, 256, 4))
    x = conv_block(inps, 1, 32)
    x = BatchNormalization(axis=1)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
    out = Dense(2,activation='softmax')(x)

    model = Model(inputs=inps, outputs=out)
    return model

def conv_block(prev_layer, layers, filters):
    for i in range(layers):
        conv = Conv2D(filters, 3, padding='same', strides=(1,1),activation="relu")(prev_layer)
        prev_layer = BatchNormalization(axis=1)(conv)
    return AveragePooling2D(pool_size=(2,2), strides=(2,2) )(prev_layer)
