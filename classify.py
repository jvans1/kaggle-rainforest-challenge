from image_loader import CSVLabelsToOneHot, DirectoryIterator, load_to_numpy
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

class LossHistory(Callback):
    def on_train_beg(self, logs={}):
        print("On Train Beg")

    def on_batch_end(self, batch, logs={}):
        if logs['acc'] < 0.1:
            super
            print("\r\nDramatic Drop", logs['acc'])
        else:
            super

history = LossHistory()

output = CSVLabelsToOneHot(classifications, 'train_v2.csv').generate_one_hot()

batch_size = 32
gen = DirectoryIterator('sample/train', output, len(classifications), batch_size=batch_size)
val_gen = DirectoryIterator('sample/valid', output, len(classifications), batch_size=batch_size)

inps = Input(shape=(4, 256, 256))
xinps = BatchNormalization()(inps)
flattened = Flatten()(xinps)
hidden = Dense(500, activation='relu')(flattened)
hidden = BatchNormalization()(hidden)
outs = [ Dense(1,activation='sigmoid')(hidden) for _ in classifications]

model = Model(inputs=inps, outputs=outs)

model.compile(optimizer=Adam(lr=0.0000001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(gen, 3000 / batch_size, epochs=8, validation_data=val_gen, validation_steps=2000 / batch_size)

