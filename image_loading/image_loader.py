import csv
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

def train_generator(settings):
    output = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')
    return DirectoryIterator(settings.train_folder, output, len(settings.training_classes), settings.training_filenames, settings=settings,batch_size=settings.batch_size)

def validation_generator(settings):
    output = classify_images(settings.training_classes, 'train_v2.csv', 'jpg')
    return DirectoryIterator(settings.validation_folder, output, len(settings.training_classes), settings.validation_filenames, settings=settings,batch_size=settings.batch_size)

def evaluation_validation_data(settings):
    filenames = os.listdir(settings.validation_folder)
    return filenames, DirectoryIterator(settings.validation_folder, None, len(settings.training_classes), settings.validation_filenames, settings=settings,batch_size=settings.batch_size, shuffle=False)

def evaluation_data(settings):
    filenames = os.listdir(settings.train_folder)
    return filenames, DirectoryIterator(settings.train_folder, None, len(settings.training_classes), settings.training_filenames, batch_size=settings.batch_size, shuffle=False)

def load_to_numpy(path):
    img = load_img(path)
    return img_to_array(img)

def load_img(absolute_path):
    img = Image.open(absolute_path)
    return img

def img_to_array(img):
    x = np.asarray(img, dtype=K.floatx())
    return x

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


def classify_images(classifications, filename, img_format):
    images = {}
    with open(filename) as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            i += 1
            if i == 1: continue
            images[row[0]+"."+img_format] = one_hot(row[1].split(' '), classifications)
    return images


def one_hot_to_labels(one_hot, classifications):
    res = []
    for present, classification in zip(one_hot, classifications):
        if present > 0.5:
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
    def __init__(self, directory,
                filename_to_binary_result_array,
                output_size, filenames,
                 target_size=(256, 256), 
                 batch_size=32, shuffle=True, seed=None,
                 settings = None,
                 follow_links=False):
        self.directory = directory
        #  self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape =  target_size + (4,)
        self.settings = settings
        self.filenames = filenames
        self.nb_sample = len(self.filenames)
        self.output_size = output_size
        self.filename_to_binary_result_array = filename_to_binary_result_array
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = []
        filenames = []
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_to_numpy(os.path.join(self.directory, fname))
            #  x = self.image_data_generator.random_transform(x)
            #  x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            if self.filename_to_binary_result_array:
                batch_y.append(self.filename_to_binary_result_array[fname])

        if self.filename_to_binary_result_array:
            outs = np.stack(np.asarray(batch_y), axis=1)
            #  outs = [ np.asarray(out) for out, i in zip(outs, range(len(outs))) if i in self.settings.measure_classes_indexes ]
            return (batch_x, list(outs))
        else:
            return batch_x
