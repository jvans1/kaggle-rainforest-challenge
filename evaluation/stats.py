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
#"agriculture"12315,
#"artisinal_mine" 339,
#"bare_ground"862,
#"blooming"332,
#"blow_down" 101,
#"clear"28431,
#"cloudy" 2089,
#"conventional_mine" 100,
#"cultivation" 4547,
#"habitation" 3660,
#"haze" 2697,
#"partly_cloudy" 7261,
#"primary" 37513,
#"road" 8071,
#"selective_logging" 340,
#"slash_burn" 209,
#"water" 7411

def labels_from_predictions(predictions):
     preds = np.concatenate(np.asarray(predictions), axis=1)
     return [ one_hot_to_labels(hot, classifications) for hot in preds ]

def compute_results(filenames, predictions, fname ='results.csv'):
    filenames = [ filename[:-4] for filename in filenames ]
    labels = labels_from_predictions(predictions)
    results = zip(filenames, labels)
    with open('results/'+fname,'w') as f:
       np.savetxt(f, results, delimiter=",", fmt="%s, %s")
