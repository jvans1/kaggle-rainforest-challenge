from image_loader import evaluation_data, train_generator, classify_images
import os
from stats import compute_results
import pdb
from model import vgg16bn
import numpy as np
import pdb
from evaluation_settings import EvaluationSettings, SampleTrainingSettings, SampleTrainingSettings

settings = SampleTrainingSettings(batch_size=40)

filenames, gen = evaluation_data(settings)
model = vgg16bn(settings.training_classes, weights_file='weights/weights.00-24.87.hdf5')

preds = model.predict_generator(gen, settings.training_batch_count, verbose=1)

compute_results(filenames, preds)
