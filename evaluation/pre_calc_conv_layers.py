from image_loading.image_loader import train_generator, classify_images, validation_generator, evaluation_validation_data, eager_load_data
from image_loading.utils import save_array, load_array
from keras.optimizers import Adam
import os
import datetime
from stats import compute_results_with_validation
import pdb
from models.model import vgg16bn
import numpy as np
import pdb
from config import SettingsWithValidation
from keras.applications import VGG16

settings = SettingsWithValidation("data/sample/train", "data/sample/valid",batch_size=4)

gen = train_generator(settings, None)

classification_map = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')
model = vgg16bn(settings.training_classes, weights_file="weights/no_top.hd5", include_top=False)

generator = train_generator(settings)
preds =  model.predict_generator(generator, settings.training_batch_count, verbose=1)

pdb.set_trace()

save_array("results/conv_training_output", preds)
save_array("results/conv_training_output_results", generator.accumulated_results)

generator = validation_generator(settings)
preds =  model.predict_generator(generator, settings.validation_batch_count, verbose=1)
save_array("results/conv_validation_output", preds)
save_array("results/conv_validation_output_results", generator.accumulated_results)

print("Saved arrays, done!")
