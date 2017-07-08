import csv
from skimage import io
import pdb
from keras import backend as K
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import Iterator

CLASSES = [
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

def train_generators(settings, image_processor=None, mapping=None):
    if mapping is None:
        mapping = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')
    train_gen = DirectoryIterator(
            settings.train_folder,
            settings.training_filenames,
            one_hot_filename_mapping=mapping,
            image_processor=image_processor,
            batch_size=settings.batch_size)
    val_gen = validation_generator(settings, mapping=mapping)
    return train_gen, val_gen

def validation_generator(settings, mapping=None):
    if mapping is None:
        mapping = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')
    return DirectoryIterator(settings.validation_folder, settings.validation_filenames, one_hot_filename_mapping=mapping,  batch_size=settings.batch_size)

def evaluation_data(settings, mapping=None):
    return settings.filenames, DirectoryIterator(settings.folder, settings.filenames, one_hot_filename_mapping=mapping, batch_size=settings.batch_size, shuffle=False)

def load_to_numpy(path):
    return io.imread(path)

def images_by_classifications():
    images = {}
    with open("train_v2.csv") as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            i += 1
            if i == 1: continue
            for classification in row[1].split(' '):
                if classification in images:
                    images[classification].append(row[0])
                else:
                    images[classification] = [row[0]]
    return images


def image_label_mapping():
    images = {}
    with open("/home/ubuntu/fastai/deeplearning1/nbs/amazon/train_v2.csv") as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            i += 1
            if i == 1: continue
            images[row[0]+"."+'jpg'] = row[1].split(' ')
    return images

def images_to_class_mapping(img_format="tif", filename="train_v2.csv"):
    images = {}
    with open(filename) as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            i += 1
            if i == 1: continue
            images[row[0]+"."+img_format] = row[1].split(' ')
    return images

def classify_images(classifications, filename, img_format):
    images = {}
    for k, v in images_to_class_mapping(img_format=img_format,filename=filename):
        images[k] =  one_hot(row[1].split(' '), classifications)


#arbitrary softmax thresholds values, calculated from training set
thresholds = [0.45972371, 0.1441146, 0.10154737, 0.041092422, 0.0097038513, 0.67270833, 0.28686687, 0.063933417, 0.23502705, 0.26686081, 0.32417396, 0.27254748, 0.72714484, 0.42341909, 0.050645847, 0.027057281, 0.28689763]

def one_hot_to_labels(one_hot, classifications):
    res = []
    for present, classification, threshold in zip(one_hot, classifications, thresholds):
        if present > threshold:
            res.append(classification)

    return ' '.join(res)

def one_hot(row, classifications):
    hot = []
    for classification in classifications:
        if classification in row:
            hot.append(1)
        else:
            hot.append(0)
    return hot

def eager_load_data(directory, result_mapping):
    data = []
    results = []
    for f in os.listdir(directory):
        data.append(load_to_numpy(os.path.join(directory, f)))
        results.append(result_mapping[f])
    return data, results

class DirectoryIterator(Iterator):
    def __init__(self,
                directory,
                filenames,
                one_hot_filename_mapping=None,
                target_size=(256, 256),
                image_processor = None,
                batch_size=32, shuffle=True, seed=None,
                follow_links=False):
        self.directory = directory
        self.image_processor = image_processor
        self.target_size = tuple(target_size)
        self.image_shape =  target_size + (4,)
        self.filenames = filenames
        self.nb_sample = len(self.filenames)
        self.one_hot_filename_mapping = one_hot_filename_mapping
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def add_result(self, output):
        self.accumulated_results.append(output)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = []
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_to_numpy(os.path.join(self.directory, fname))
            if self.image_processor:
                x = self.image_processor.random_transform(x)
            batch_x[i] = x
            if self.one_hot_filename_mapping:
                res = self.one_hot_filename_mapping[fname]
                batch_y.append(res)

        if len(batch_y) > 0:
            outs = np.asarray(batch_y)
            return (batch_x, outs)
        else:
            return batch_x
