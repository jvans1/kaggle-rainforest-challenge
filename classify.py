from image_loader import CSVLabelsToOneHot, DirectoryIterator, load_to_numpy
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

labels = CSVLabelsToOneHot(classifications, 'train_v2.csv').generate_one_hot()

batch_size = 32
gen = DirectoryIterator('sample/train', labels, len(classifications), batch_size=batch_size)
val_gen = DirectoryIterator('sample/valid', labels, len(classifications), batch_size=batch_size)

inps = Input(shape=(4, 256, 256))
xinps = BatchNormalization()(inps)
flattened = Flatten()(xinps)
hidden = Dense(500, activation='relu')(flattened)
hidden = BatchNormalization()(hidden)
out = Dense(len(classifications),activation='sigmoid')(hidden)
model = Model(inps, out)
model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(gen, 1500 / batch_size, epochs=10, validation_data=val_gen, validation_steps= 500 / batch_size)

#  filename = 'train_19419.jpg'
#  img = load_to_numpy('sample/valid/'+filename)
#  result = labels[filename]
#  predictions = model.predict(np.asarray([img]))
#  print(result)
#  print(predictions[0])

