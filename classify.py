from image_loader import classify_images, DirectoryIterator, load_to_numpy, one_hot_to_labels
import os
import pdb
from keras.callbacks import Callback
from stats import compase_results_with_train_labels, compute_result
from model import vgg16bn
import numpy as np
import pdb
from evaluation_settings import SampleTrainingSettings


settings = SampleTrainingSettings()

output = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')

gen = DirectoryIterator(settings.train_folder, output, len(settings.training_classes), settings.training_filenames, batch_size=settings.batch_size)
val_gen = DirectoryIterator(settings.validation_folder, output, len(settings.training_classes), settings.validation_filenames, batch_size=settings.batch_size)


model = vgg16bn(settings.training_classes, weights_file="weights/weights.00-6.34.hdf5")
model.fit_generator(gen,  settings.training_batch_count, epochs=20, validation_data=val_gen,  validation_steps=settings.validation_batch_count, callbacks=settings.callbacks)


#  dirname = 'train/train'
#  filenames = os.listdir(dirname)

#  pred = DirectoryIterator(dirname, None, len(classifications), filenames, batch_size=batch_size, shuffle=False)

#  preds = model.predict_generator(pred, (len(filenames) / batch_size) + 1, verbose=1)
#  results = np.concatenate(np.asarray(preds), axis=1)
#  compute_result(filenames, results)

