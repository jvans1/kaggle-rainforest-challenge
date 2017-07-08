from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout
from image_loading.image_loader import eager_load_data
from keras.regularizers import l2
from image_loading.utils import save_array, load_array

from keras.optimizers import Adam
import numpy as np
from keras.applications import VGG16
import pdb
from keras.models import Model
def linear_model(output_classes):
    inps = Input(shape=(256, 256, 4))
    bn_inp = BatchNormalization(axis=1)(inps)
    flattened = Flatten()(bn_inp)
    dense = Dense(1024, activation='relu')(flattened)
    bn_out = BatchNormalization(axis=1)(dense)
    outs = [ Dense(1,activation='sigmoid', name=classification)(bn_out) for classification in output_classes]

    model = Model(inputs=inps, outputs=outs)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
