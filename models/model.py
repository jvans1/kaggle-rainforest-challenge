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
    inps = Input(shape=(256,256, 3))
    vgg = VGG16(input_shape=(256, 256,3), include_top=False, weights="imagenet")
    return vgg

def conv_block(prev_layer, layers, filters):
    for i in range(layers):
        conv = Conv2D(filters, 3, padding='same', strides=(1,1),activation="relu")(prev_layer)
        prev_layer = BatchNormalization(axis=1)(conv)
    return MaxPooling2D(pool_size=(2,2), strides=(2,2) )(prev_layer)

def dense_layers(dropout=0.0, regularization=0.0):
    return [
        Flatten(input_shape=(8,8,512)),
        BatchNormalization(),
        Dense(4096, activation="relu", kernel_regularizer=l2(regularization)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(4096, activation="relu", kernel_regularizer=l2(regularization)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(4, activation="softmax"),
    ]

def full_model(weights_file=None, dropout=0.0, regularization=0.0):
    inps = Input(shape=(256,256, 3))
    vgg = VGG16(input_shape=(256, 256,3), include_top=False, weights="imagenet")
    model = Sequential(vgg.layers + dense_layers(dropout, regularization))
    if weights_file is not None:
      model.load_weights(weights_file)
    return model
