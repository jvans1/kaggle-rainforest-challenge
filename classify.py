from image_loader import classify_images, eager_load_data
import os
import pdb
from keras.callbacks import Callback
from stats import compase_results_with_train_labels, compute_result
from model import vgg16bn, only_conv_layers_model, only_dense_model, load_train_conv_output, load_valid_conv_output, save_conv_output
import numpy as np
import pdb
from evaluation_settings import SampleTrainingSettings

settings = SampleTrainingSettings(batch_size=16)

output = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')

model = only_conv_layers_model(settings.training_classes)
save_conv_output(model, settings, output)

model = only_dense_model(settings.training_classes)

val_out, val_results = load_valid_conv_output()
train_out, train_results = load_train_conv_output()
pdb.set_trace()
#  model.fit(train_out, train_results, validation_data=(val_out, val_results))
