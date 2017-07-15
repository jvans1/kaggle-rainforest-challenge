import os
import numpy as np
from image_loading.image_loader import images_by_classifications
from shutil import copyfile, rmtree


base = 'data/samples'
paths = ['/partly_cloudy']

for path in paths:
    full_path = base+path
    if os.path.exists(full_path):
        rmtree(full_path)
        os.mkdir(full_path)
        os.mkdir(full_path+'/valid')
        os.mkdir(full_path+'/train')
    else:
        os.mkdir(full_path)
        os.mkdir(full_path+'/valid')
        os.mkdir(full_path+'/train')


    train_files = os.listdir('data'+path+'/train')
    valid_files = os.listdir('data'+path+'/valid')

    train_sample = train_files[:1200]
    valid_sample = valid_files[:800]

    for f in train_sample:
        copyfile('data/train-jpg/'+f, full_path+'/train/'+f)

    for f in valid_sample:
        copyfile('data/train-jpg/'+f, full_path+'/valid/'+f)
