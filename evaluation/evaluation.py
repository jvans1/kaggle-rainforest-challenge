from image_loading.image_loader import evaluation_data
from keras.optimizers import Adam
import pdb
import os
import numpy as np
from config import PredictSettings
from models.specializer import specializer
from image_loading.utils import binary_one_hot_mapping

settings = PredictSettings("data/train-jpg", batch_size=64)
mapping = binary_one_hot_mapping("cloud")

filenames, gen = evaluation_data(settings, mapping=mapping)
model = specializer("cloudy", 0.0)
model.load_weights("weights/cloudy-sample-1")

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
preds = model.evaluate_generator(gen, settings.batch_count)
pdb.set_trace()
