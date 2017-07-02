from image_loading.image_loader import evaluation_data, train_generator, classify_images, evaluation_validation_data
import os
from stats import compute_results
import pdb
from models.model import vgg16bn
import numpy as np
import pdb
from config import SampleTrainingSettings

settings = SampleTrainingSettings(batch_size=40)

filenames, gen = evaluation_data(settings)

val_filenames, val_gen = evaluation_validation_data(settings)
model = vgg16bn(settings.training_classes, weights_file=settings.weights_file)

preds = model.evaluate_generator(gen, settings.training_batch_count, validation_data=(val_filenames, val_gen), validation_steps=settings.validation_batch_count)

compute_results(filenames, preds)
