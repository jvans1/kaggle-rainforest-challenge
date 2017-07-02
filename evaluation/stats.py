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
from image_loader import  one_hot_to_labels

def compute_results_with_validation(filenames, predictions, mapping):
    labels = labels_from_predictions(predictions)
    fnames = [ filename[:-4] for filename in filenames ]
    actual_labels = [ one_hot_to_labels(mapping[filename], classifications) for filename in filenames ]
    results = zip(fnames, labels, actual_labels)
    with open('results.csv','w') as f:
       np.savetxt(f, results, delimiter=",", fmt="%s, %s, %s")



def labels_from_predictions(predictions):
     preds = np.concatenate(np.asarray(predictions), axis=1)
     return [ one_hot_to_labels(hot, classifications) for hot in preds ]

def compute_results(filenames, predictions):
    filenames = [ filename[:-4] for filename in filenames ]
    labels = labels_from_predictions(predictions)
    results = zip(filenames, labels)
    with open('results.csv','w') as f:
       np.savetxt(f, results, delimiter=",", fmt="%s, %s")

