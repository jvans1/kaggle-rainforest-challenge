from image_loader import classify_images, eager_load_data, train_generator, validation_generator
import os
import pdb
from keras.callbacks import Callback
from stats import compase_results_with_train_labels, compute_result
from model import vgg16bn, only_conv_layers_model, only_dense_model, load_train_conv_output, load_valid_conv_output, save_conv_output
import numpy as np
import pdb
from evaluation_settings import SampleTrainingSettings, TrainingSettings

settings = TrainingSettings(batch_size=40)


train_gen = train_generator(settings)
val_gen = validation_generator(settings)
model = vgg16bn(settings.training_classes, train_conv_layers=True, weights_file='weights.00-24.87.hdf5')
model.fit_generator(train_gen, settings.training_batch_count, epochs=50, validation_data = val_gen, validation_steps=settings.validation_batch_count, callbacks=settings.callbacks)
