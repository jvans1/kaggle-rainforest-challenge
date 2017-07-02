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
def compase_results_with_train_labels(prefix, results, mapping):
    exact_mappings = 0
    for fname, prediction in results:
        prediction = one_hot_to_labels(prediction, classifications)
        actual     = one_hot_to_labels(mapping[fname], classifications)
        if actual == prediction:
            exact_mappings +=1
            print("Correctly predicted: "+prediction)
        else:
            print("Predicted: "+prediction)
            print("was " + actual +"\r\n")
    print("Total exact mappings: " + str(exact_mappings))


def compute_result(filenames, predictions):
    labels = [ one_hot_to_labels(hot, classifications) for hot in predictions ]
    filenames = [ filename[:-4] for filename in filenames ]
    results = zip(filenames, labels)
    with open('results.csv','w') as f:
       np.savetxt(f, results, delimiter=",", fmt="%s, %s")

