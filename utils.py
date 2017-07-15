import csv
import os
from image_loader import load_to_numpy, DirectoryIterator
import bcolz
from matplotlib import pyplot as plt

#counts
#  {
#    LT 1000
#   'selective_logging': 340,
#    'conventional_mine': 100,
#    'slash_burn': 209,
#    'artisinal_mine': 339,
#    'blooming': 332,
#    'bare_ground': 862,
#    'blow_down': 101,

#    GT 1000
#    'cultivation': 4547,
#    'habitation': 3660,
#    'primary': 37513,
#    'water': 7411,
#    'agriculture': 12315,
#    'road': 8071}

#    Atmospheric
#    'cloudy': 2089,
#    'haze': 2697,
#    'partly_cloudy': 7261,
#    'clear': 28431,
#  116278

#train_data = 34816


def plot(path):
    img = load_to_numpy(path)
    plt.imshow(img)


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]


def atmospheric_mapping():
    labels = ['cloudy', 'partly_cloudy', 'haze', 'clear']
    mapping = {}
    for fname in images_to_class:
        mapping[fname] = [0, 0, 0, 0]
        for label, index in zip(labels, range(len(labels))):
            if label in images_to_class[fname]:
                mapping[fname][index] = 1
    return mapping

def binary_one_hot_mapping(klass):
    mapping = {}
    for fname in images_to_class:
        if klass in images_to_class[fname]:
            mapping[fname] = [0, 1]
        else:
            mapping[fname] = [1, 0]
    return mapping

def incorrect_predictions(results):
    wrong = []
    mapping = images_to_class_mapping()
    for fname, pred in results:
        one_hot = mapping[fname]
        if pred[0] > 0.50 and one_hot[0] == 0:
            wrong.append((fname, pred))
        elif pred[1] > 0.50 and one_hot[1] == 0:
            wrong.append((fname, pred))
    return wrong

def images_to_class_mapping(img_format="jpg", filename="train_v2.csv"):
    images = {}
    with open(filename) as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            i += 1
            if i == 1: continue
            images[row[0]+"."+img_format] = row[1].split(' ')
    return images

images_to_class = images_to_class_mapping()
