from image_loader import classify_images, DirectoryIterator, load_to_numpy, one_hot_to_labels
import os
from model import create_model
import pdb
from keras.callbacks import Callback
from stats import compase_results_with_train_labels
from jvVgg16bn import vgg16bn
from keras.optimizers import Adam
import numpy as np
import pdb

classifications = [
    "agriculture",
    "artisinal_mine",
    "bare_ground",
    "blooming",
    "blow_down",
    "clear",
    "cloudy",
    "conventional_mine",
    "cultivation",
    "habitation",
    "haze",
    "partly_cloudy",
    "primary",
    "road",
    "selective_logging",
    "slash_burn",
    "water"
]


output = classify_images(classifications, 'train_v2.csv', 'jpg')
batch_size = 32
train_filenames =  os.listdir('sample/train')
valid_filenames = os.listdir('sample/valid')
gen = DirectoryIterator('sample/train', output, len(classifications), train_filenames, batch_size=batch_size)
val_gen = DirectoryIterator('sample/valid', output, len(classifications), valid_filenames, batch_size=batch_size)


input_size = 3000

model = vgg16bn(classifications)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(gen, input_size / batch_size, epochs=8, validation_data=val_gen, validation_steps=2000 / batch_size)

model.save_weights("8_epochs_vgg")

valid_filenames = os.listdir('sample/valid')
pred = DirectoryIterator('sample/valid', output, len(classifications), valid_filenames, batch_size=batch_size, shuffle=False)

preds = model.predict_generator(pred, input_size / batch_size)
results = np.concatenate(np.asarray(preds), axis=1)

compase_results_with_train_labels('sample/valid', zip(valid_filenames, results), output)
