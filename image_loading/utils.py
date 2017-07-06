import bcolz
from image_loading.image_loader import images_to_class_mapping
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

images_to_class = images_to_class_mapping()

def binary_one_hot_mapping(klass):
    mapping = {}
    for fname in images_to_class:
        if klass in images_to_class[fname]:
            mapping[fname] = [0, 1]
        else:
            mapping[fname] = [1, 0]
    return mapping
