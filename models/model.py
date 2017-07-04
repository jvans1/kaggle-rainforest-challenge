from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout
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
        prev_layer = conv #BatchNormalization(axis=1)(conv)
    return MaxPooling2D(pool_size=(2,2), strides=(2,2) )(prev_layer)


def vgg16bn(output_classes, weights_file=None, include_top=True, train_conv_layers= False, trainable=True):
    inputs = Input(shape=(256, 256, 4))
    inps = BatchNormalization(axis=1)(inputs)
    cb = conv_block(inps, 2, 64)
    cb = conv_block(cb, 2, 128)
    cb = conv_block(cb, 3, 256)
    cb = conv_block(cb, 3, 512)
    cb = conv_block(cb, 3, 512)
    flattened = Flatten()(cb)

    dense_1 = Dense(2048, activation='relu')(flattened)
    bn_dense_1 = BatchNormalization(axis=1)(dense_1)
    dropout_1 = Dropout(0.1)(bn_dense_1)

    dense_2 = Dense(2048, activation='relu')(dropout_1)
    bn_dense_2 = BatchNormalization(axis=1)(dense_2)
    dropout_2 = Dropout(0.1)(bn_dense_2)

    outs = [ Dense(1,activation='sigmoid', name=classification)(dropout_2) for classification in output_classes]
    model = Model(inputs=inputs,outputs=outs)
    model.summary()

    if not trainable:
        for layer in model.layers:
            layer.trainable = False

    if weights_file:
        model.load_weights(weights_file)

    if not include_top:
        for i in range(24):
            model.layers.pop()
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
