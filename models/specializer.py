from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Concatenate
import re
from image_loading.image_loader import eager_load_data
from keras.regularizers import l2
from image_loading.utils import save_array, load_array

from keras.optimizers import Adam
import numpy as np
from keras.applications import VGG16
import pdb
from keras.models import Model

def conv_block(prev_layer, layers, filters):
    for i in range(layers):
        conv = Conv2D(filters, 3, padding='same', strides=(1,1),activation="relu")(prev_layer)
        prev_layer = BatchNormalization(axis=1)(conv)
    return MaxPooling2D(pool_size=(2,2), strides=(2,2) )(prev_layer)


def specializer(name, dropout_rate):
    image_input = Input(shape=(256, 256, 4), name=name)
    inps = BatchNormalization(axis=1)(image_input)
    cb = conv_block(inps, 2, 64)
    cb = conv_block(cb, 2, 128)
    cb = conv_block(cb, 3, 256)
    cb = conv_block(cb, 3, 512)
    cb = conv_block(cb, 3, 512)
    flattened = Flatten()(cb)

    dense_1 = Dense(4096, activation='relu')(flattened)
    bn_dense_1 = BatchNormalization(axis=1)(dense_1)
    dropout_1 = Dropout(dropout_rate)(bn_dense_1)

    dense_2 = Dense(4096, activation='relu')(dropout_1)
    bn_dense_2 = BatchNormalization(axis=1)(dense_2)
    dropout_2 = Dropout(dropout_rate)(bn_dense_2)
    out = Dense(2, activation='softmax')(dropout_2)

    return Model(inputs=image_input, outputs=out)
