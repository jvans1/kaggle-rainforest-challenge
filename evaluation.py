from image_loader import evaluation_data
import os
import pdb
from stats import compute_result
from model import vgg16bn
import numpy as np
import pdb
from evaluation_settings import EvaluationSettings

settings = EvaluationSettings(batch_size=32)

filenames, gen = evaluation_data(settings)
model = vgg16bn(settings.training_classes, weights_file='weights/weights.00-24.87.hdf5')

preds = model.predict_generator(gen, settings.batch_count)

compute_results(filenames, preds)
