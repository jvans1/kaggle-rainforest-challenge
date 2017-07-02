from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout
from image_loader import eager_load_data
from keras.regularizers import l2
from utils import save_array, load_array

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

def fc_block(prev_layer):
    return Dense(4096, activation='relu')(prev_layer)

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


def load_train_conv_output():
    output = load_array('train_conv_output_array')
    results = load_array('train_conv_output_array_results')
    return output, results

def load_valid_conv_output():
    output = load_array('valid_conv_output_array')
    results = load_array('valid_conv_output_array_results')
    return output, results



def save_conv_output(model, settings, output):
    data, results = eager_load_data(settings.train_folder, output)
    train_conv_output = model.predict(np.asarray(data))
    pdb.set_trace()

    save_array('train_conv_output_array', train_conv_output)
    save_array('train_conv_output_array_results', results)

    data, results = eager_load_data(settings.validation_folder, output)
    valid_conv_output = model.predict(np.asarray(data))

    save_array('valid_conv_output_array', valid_conv_output)
    save_array('valid_conv_output_array_results', results)



def vgg16bn(output_classes, weights_file=None, train_conv_layers= False):
    model = VGG16(include_top=False, weights='imagenet')

    inps = Input(shape=(256, 256, 4))
    bn_inp = BatchNormalization(axis=1)(inps)
    first_conv_layer = Conv2D(64, 2, padding='same', strides=(1,1), activation='relu')(bn_inp)

    prev = first_conv_layer
    for layer in model.layers[2:len(model.layers) -1]:
        prev = layer(prev)

    flattened = Flatten()(prev)
    dense_1 = Dense(2048, activation='relu', name="First Dense Layer", kernel_regularizer=l2(0.005)   )(flattened)
    bn_dense_1 = BatchNormalization(axis=1)(dense_1)
    dropout_1 = Dropout(0.4)(bn_dense_1)

    dense_2 = Dense(2048, activation='relu', name="Last Dense Layer", kernel_regularizer=l2(0.005)  )(dropout_1)
    bn_dense_2 = BatchNormalization(axis=1)(dense_2)
    dropout_2 = Dropout(0.4)(bn_dense_2)
    outs = [ Dense(1,activation='sigmoid', name=classification)(dropout_2) for classification in output_classes]
    model = Model(inputs=inps, outputs=outs) 

    if not train_conv_layers:
        for layer in model.layers[:CONV_INDEX]:
            layer.trainable = False

    if weights_file:
        model.load_weights(weights_file)

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
