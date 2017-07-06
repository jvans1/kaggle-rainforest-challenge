from keras.optimizers import Adam
import numpy as np
from stats import compute_results
from image_loading.image_loader import DirectoryIterator
import pdb
from config import PredictSettings
from image_loading.image_loader import train_generators
from models.specializer import specializer

settings = PredictSettings("data/samples/cloudy/valid")
gen = DirectoryIterator(settings.folder, settings.filenames, batch_size=64, shuffle=False)
filenames = settings.filenames

model = specializer("cloudy", 0.0)
model.load_weights('weights/cloudy-weights.04-0.79.hdf5')

[ l.set_weights([ w / 3.33 for w in l.get_weights()]) for l in model.layers ]


model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
preds = model.predict_generator(gen, settings.batch_count, verbose=1)

filenames = [ filename[:-4] for filename in filenames ]
labels = []
pdb.set_trace()
for one_hot in preds:
    if one_hot[1] > 0.5:
        labels.append("cloudy")
    else:
        labels.append("other")

results = zip(filenames, labels)
with open('results/cloudy-results','w') as f:
   np.savetxt(f, results, delimiter=",", fmt="%s, %s")
