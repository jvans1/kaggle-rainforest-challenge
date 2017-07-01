from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
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

def vgg16bn(output_classes, weights_file=None):
    model = VGG16(include_top=False, weights='imagenet')

    inps = Input(shape=(256, 256, 4))
    bn_inp = BatchNormalization(axis=1)(inps)
    first_conv_layer = Conv2D(64, 2, padding='same', strides=(1,1), activation='relu')(bn_inp)

    prev = first_conv_layer
    for layer in model.layers[2:len(model.layers) -1]:
        prev = layer(prev)

    flattened = Flatten()(prev)
    dense_1 = Dense(4096, activation='relu' )(flattened)
    bn_dense_1 = BatchNormalization(axis=1)(dense_1)

    dense_2 = Dense(4096, activation='relu' )(bn_dense_1)
    bn_dense_2 = BatchNormalization(axis=1)(dense_2)
    outs = [ Dense(1,activation='sigmoid', name=classification)(bn_dense_2) for classification in output_classes]
    model = Model(inputs=inps, outputs=outs) 
    for layer in model.layers[:-21]:
        layer.trainable = False

    if weights_file:
        model.load_weights(weights_file)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
