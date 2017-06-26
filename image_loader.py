import csv
from keras import backend as K
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import Iterator


class CSVLabelsToOneHot(object):
    def __init__(self, classifications, filename):
        self.classifications = classifications
        self.filename = filename

    def generate_one_hot(self, im_format="jpg"):
        images = {}
        with open(self.filename) as csvfile:
            i = 0
            reader = csv.reader(csvfile)
            for row in reader:
                i += 1
                if i == 1: continue
                images[row[0]+"."+im_format] = self.__one_hot(row[1].split(' '))
        return images

    def __one_hot(self, row):
        hot = np.zeros(len(self.classifications))
        for r in row:
            hot[self.classifications.index(r)] = 1
        return hot


class DirectoryIterator(Iterator):
    def __init__(self, directory,
                class_indices,
                 target_size=(256, 256), 
                 batch_size=32, shuffle=True, seed=None,
                 follow_links=False):
        self.directory = directory
        #  self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = (4,) + target_size

        # first, count the number of samples and classes

        self.filenames = os.listdir(directory)
        self.nb_sample = len(self.filenames)
        self.nb_class = len(class_indices)
        self.class_indices = class_indices
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = ['' for _, _ in enumerate(index_array) ]
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = self.__load_img(os.path.join(self.directory, fname), self.target_size)
            x = self.__img_to_array(img)
            #  x = self.image_data_generator.random_transform(x)
            #  x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.class_indices[fname]
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        return batch_x, batch_y

    def __load_img(self, absolute_path, dimensions):
        img = Image.open(absolute_path)
        #img.resize(dimensions[1], dimensions[0])
        return img
    def __img_to_array(self, img):
        x = np.asarray(img, dtype=K.floatx())
        return x.transpose(2, 0, 1)
