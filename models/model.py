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
        prev_layer = BatchNormalization(axis=1)(conv)
    return MaxPooling2D(pool_size=(2,2), strides=(2,2) )(prev_layer)

CONV_INDEX = 20
def only_conv_layers_model(classes):
    model = vgg16bn(classes)
    model.layers = model.layers[:CONV_INDEX]
    return model

def only_dense_model(classes):
    model = vgg16bn(classes)
    model.layers = model.layers[20:]
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def vgg16bn(output_classes, weights_file=None, train_conv_layers= False, trainable=True):
    inputs = Input(shape=(256, 256, 4))
    inps = BatchNormalization(axis=1)(inputs)
    cb = conv_block(inps, 2, 64)
    cb = conv_block(cb, 2, 128)
    cb = conv_block(cb, 3, 256)
    cb = conv_block(cb, 3, 512)
    cb = conv_block(cb, 3, 512)
    flattened = Flatten()(cb)
    dense_1 = Dense(8192, activation='relu')(flattened)
    bn_dense_1 = BatchNormalization(axis=1)(dense_1)
    dropout_1 = Dropout(0.0)(bn_dense_1)

    dense_2 = Dense(8192, activation='relu')(dropout_1)
    bn_dense_2 = BatchNormalization(axis=1)(dense_2)
    dropout_2 = Dropout(0.0)(bn_dense_2)
    outs = [ Dense(1,activation='sigmoid', name=classification)(dropout_2) for classification in output_classes]
    model = Model(inputs=inputs,outputs=outs)
    model.summary()

    if not trainable:
        for layer in model.layers:
            layer.trainable = False

    if weights_file:
        model.load_weights(weights_file)

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
