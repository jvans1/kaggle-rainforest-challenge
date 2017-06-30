from image_loader import classify_images, DirectoryIterator, load_to_numpy, one_hot_to_labels
import os
import pdb
from keras.callbacks import Callback
from keras.layers import Input, Dense, Flatten, BatchNormalization
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
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

#  vgg = Vgg16BN(size=(256,256) )

inps = Input(shape=(4, 256, 256))
xinps = BatchNormalization(axis=1)(inps)
flattened = Flatten()(xinps)
hidden = Dense(500, activation='relu')(flattened)
hidden = BatchNormalization()(hidden)
outs = [ Dense(1,activation='sigmoid', name=classification)(hidden) for classification in classifications]

model = Model(inputs=inps, outputs=outs)

input_size = 3000
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(gen, input_size / batch_size, epochs=8, validation_data=val_gen, validation_steps=2000 / batch_size)
model.save_weights("8_epochs_1_hidden")


valid_filenames = os.listdir('sample/valid')
pred = DirectoryIterator('sample/valid', output, len(classifications), valid_filenames, batch_size=batch_size, shuffle=False)

preds = model.predict_generator(pred, input_size / batch_size)
results = np.concatenate(np.asarray(preds), axis=1)
print("Filename: ", valid_filenames[1:3])
print([ one_hot_to_labels(one_hot, classifications) for one_hot in results ][1:3])
