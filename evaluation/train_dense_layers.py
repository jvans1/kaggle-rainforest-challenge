from image_loading.image_loader import train_generator, classify_images, validation_generator, evaluation_validation_data, eager_load_data
from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from image_loading.utils import save_array, load_array
import os
import datetime
from stats import compute_results_with_validation
import pdb
from models.model import vgg16bn
import numpy as np
import pdb
from config import SettingsWithValidation
from keras.applications import VGG16

settings = SettingsWithValidation("data/sample/train", "data/sample/valid",batch_size=16 )

conv_features = load_array("results/conv_training_output")
conv_results  = load_array("results/conv_training_output_results")
conv_results = list(np.reshape(conv_results, (conv_results.shape[1], conv_results.shape[0])))
val_features  = load_array("results/conv_validation_output")
val_results   = load_array("results/conv_validation_output_results")
val_results = list(np.reshape(val_results, (val_results.shape[1], val_results.shape[0])))

print(conv_features.shape)
inps = Input( shape=(8,8,512) )
pool = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(inps)
flattened = Flatten()(pool)
dense_1 = Dense(2048, activation='relu')(flattened)
bn_dense_1 = BatchNormalization(axis=1)(dense_1)
dropout_1 = Dropout(0.0)(bn_dense_1)

dense_2 = Dense(2048, activation='relu')(dropout_1)
bn_dense_2 = BatchNormalization(axis=1)(dense_2)
dropout_2 = Dropout(0.0)(bn_dense_2)

outs = [ Dense(1,activation='sigmoid', name=classification)(dropout_2) for classification in settings.training_classes]

model = Model(inputs=inps,outputs=outs)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(conv_features, conv_results, validation_data=(val_features, val_results))
