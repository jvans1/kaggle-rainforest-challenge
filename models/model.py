from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout
from image_loading.image_loader import eager_load_data
from keras.models import Sequential
from keras.regularizers import l2

from keras.optimizers import Adam
import numpy as np
from keras.applications import VGG16
import pdb
from keras.models import Model
def vgg():
    inps = Input(shape=(256,256, 4))
    x = BatchNormalization()(inps)
    x = conv_block(x, 2, 64)
    x = conv_block(x, 2, 128)
    x = conv_block(x, 3, 256)
    x = conv_block(x, 3, 512)
    x = conv_block(x, 3, 512)
    x = Flatten()(x)
    x = Dense(2048,activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(2048, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(2, activation="softmax")(x)
    return Model(inputs=inps, outputs=x)

def conv_block(prev_layer, layers, filters):
    for i in range(layers):
        conv = Conv2D(filters, 3, padding='same', strides=(1,1),activation="relu")(prev_layer)
        prev_layer = BatchNormalization(axis=1)(conv)
    return MaxPooling2D(pool_size=(2,2), strides=(2,2) )(prev_layer)
