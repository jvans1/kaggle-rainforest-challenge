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

common = ['agriculture', 'cultivation', 'habitation', 'primary', 'road', 'water']
def rare_model(dropout=0.0, regularization=0.0):
    inputs = Input(shape=(256,256,3)
    x = conv_block(inputs, 2, 64)
    x = conv_block(x, 2, 128)
    x = conv_block(x, 3, 256)
    x = conv_block(x, 3, 512)
    x = conv_block(x, 3, 512)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation="relu", kernel_regularizer=l2(regularization))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(4096, activation="relu", kernel_regularizer=l2(regularization))(x)
    x = BatchNormalization()
    x = Dropout(dropout)(x)
    outs = [Dense(2, activation="softmax", name=label)(x) for label in common ]
    model = Model(inputs=inputs,outputs=outs)


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
