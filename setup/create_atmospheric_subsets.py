import os
import numpy as np
from image_loading.image_loader import images_by_classifications
from shutil import copyfile, rmtree

paths = ['data/partly_cloudy']
for path in paths:
    if os.path.exists(path):
        rmtree(path)
        os.mkdir(path)
        os.mkdir(path+'/valid')
        os.mkdir(path+'/train')
    else:
        os.mkdir(path)
        os.mkdir(path+'/valid')
        os.mkdir(path+'/train')


    images_by_class = images_by_classifications()

    partly_cloudy_files = images_by_class["partly_cloudy"]

    fifteen_percent = len(partly_cloudy_files) / 5
    random = np.random.permutation(partly_cloudy_files)
    train = random[fifteen_percent:]
    validation = random[:fifteen_percent]
    for f in train:
        copyfile('data/train-jpg/'+f+'.jpg', path+'/train/'+f+'.jpg')

    for f in validation:
        copyfile('data/train-jpg/'+f+'.jpg', path+'/valid/'+f+'.jpg')

    cloudy_files = images_by_class["cloudy"][:3000]
    haze_files = images_by_class["haze"][:3000]
    clear_files = images_by_class["clear"][:1000]
    for file_set in [cloudy_files, partly_cloudy_files, haze_files, clear_files]:
        fifteen_percent = len(file_set) / 3
        random = np.random.permutation(file_set)
        train = random[fifteen_percent:]
        validation = random[:fifteen_percent]
        for f in train:
            copyfile('data/train-jpg/'+f+'.jpg', path+'/train/'+f+'.jpg')

        for f in validation:
            copyfile('data/train-jpg/'+f+'.jpg', path+'/valid/'+f+'.jpg')
