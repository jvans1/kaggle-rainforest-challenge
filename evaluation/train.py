from image_loading.image_loader import train_generator, classify_images, validation_generator, evaluation_validation_data
from keras.preprocessing.image import ImageDataGenerator
import os
import datetime
from stats import compute_results_with_validation
import pdb
from models.model import vgg16bn
import numpy as np
import pdb
from config import SettingsWithValidation

settings = SettingsWithValidation("data/sample/train", "data/sample/valid",batch_size=16)

image_processor = ImageDataGenerator(rotation_range=180, shear_range=0.3, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)
gen = train_generator(settings, image_processor)

val_gen = validation_generator(settings)
model =  vgg16bn(settings.training_classes, weights_file=None, train_conv_layers=True) # linear_model(settings.training_classes)#
preds =  model.fit_generator(gen, settings.training_batch_count, epochs=3, validation_data = val_gen, validation_steps=settings.validation_batch_count, callbacks=settings.callbacks)

filenames, gen = evaluation_validation_data(settings)
preds = model.predict_generator(gen, settings.validation_batch_count, verbose=1)

output = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')
current_time = str(datetime.datetime.now())
compute_results_with_validation(filenames, preds,output, save_to="training-"+current_time+".csv")
