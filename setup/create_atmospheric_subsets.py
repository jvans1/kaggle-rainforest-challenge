import os
import numpy as np
from image_loading.image_loader import images_by_classifications
from shutil import copyfile, rmtree

path = 'data/haze'
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

partly_cloudy_files = images_by_class["haze"]

haze_train = []
haze_train += partly_cloudy_files
rand = np.random.permutation(haze_train)
h_train = rand[:2000]
h_valid = rand[2000:]

print("Putting " +str(len(h_train)) + " haze files in training set")

for f in h_train:
    copyfile('data/train-jpg/'+f+'.jpg', path+'/train/'+f+'.jpg')

for f in h_valid:
    copyfile('data/train-jpg/'+f+'.jpg', path+'/valid/'+f+'.jpg')


files = []
cloudy_files = images_by_class["cloudy"][:3000]
files += cloudy_files
haze_files = images_by_class["partly_cloudy"][:3000]
files += haze_files
clear_files = images_by_class["clear"][:3000]
files += clear_files
files = list(set(files))

rand = np.random.permutation(files)
train = rand[:7000]
valid = rand[7000:]

for f in train:
    copyfile('data/train-jpg/'+f+'.jpg', path+'/train/'+f+'.jpg')

for f in valid:
    copyfile('data/train-jpg/'+f+'.jpg', path+'/valid/'+f+'.jpg')
