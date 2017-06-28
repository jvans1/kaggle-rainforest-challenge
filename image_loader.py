import csv
import pdb
from keras import backend as K
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import Iterator

def load_to_numpy(path):
    img = load_img(path)
    return img_to_array(img)

def load_img(absolute_path):
    img = Image.open(absolute_path)
    return img

def img_to_array(img):
    x = np.asarray(img, dtype=K.floatx())
    return x.transpose(2, 0, 1)

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
                images[row[0]+"."+im_format] = self.__output(row[1].split(' '))
        return images

    def __output(self, row):
        hot = []
        for classification in self.classifications:
            if classification in row:
                hot.append([1])
            else:
                hot.append([0])
        return hot


class DirectoryIterator(Iterator):
    def __init__(self, directory,
                filename_to_binary_result_array,
                output_size,
                 target_size=(256, 256), 
                 batch_size=32, shuffle=True, seed=None,
                 follow_links=False):
        self.directory = directory
        #  self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = (4,) + target_size
        self.filenames = os.listdir(directory)
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
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_to_numpy(os.path.join(self.directory, fname))
            #  x = self.image_data_generator.random_transform(x)
            #  x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y.append(self.filename_to_binary_result_array[fname])
        outs = np.concatenate(np.asarray(batch_y), axis=1)
        return (batch_x, list(outs))

