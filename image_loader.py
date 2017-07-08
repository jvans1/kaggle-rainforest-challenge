from skimage import io
import os
from keras import backend as K
from keras.preprocessing.image import Iterator
import numpy as np
def load_to_numpy(path):
    return io.imread(path)

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
