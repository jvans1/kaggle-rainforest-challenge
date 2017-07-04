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
from image_loading.image_loader import  one_hot_to_labels

def compute_results_with_validation(filenames, predictions, mapping, save_to ='results.csv'):
    labels = labels_from_predictions(predictions)
    fnames = [ filename[:-4] for filename in filenames ]
    actual_labels = [ one_hot_to_labels(mapping[filename], classifications) for filename in filenames ]
    results = zip(fnames, labels, actual_labels)
    with open('results/'+save_to,'w') as f:
       np.savetxt(f, results, delimiter=",", fmt="%s, %s, %s")



#num records in each class sorted alphabetically
#counts =  [12315, 339, 862, 332, 101, 28431, 2089, 100, 4547, 3660, 2697, 7261, 37513, 8071, 340, 209, 7411]

def labels_from_predictions(predictions):
     preds = np.concatenate(np.asarray(predictions), axis=1)
     return [ one_hot_to_labels(hot, classifications) for hot in preds ]

def compute_results(filenames, predictions, fname ='results.csv'):
    filenames = [ filename[:-4] for filename in filenames ]
    labels = labels_from_predictions(predictions)
    results = zip(filenames, labels)
    with open('results/'+fname,'w') as f:
       np.savetxt(f, results, delimiter=",", fmt="%s, %s")
