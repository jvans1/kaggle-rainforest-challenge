import os
from shutil import copyfile, rmtree
from importlib import import_module
from image_loading.image_loader import images_by_classifications

if not os.path.exists('data/train-jpg'):
    raise Exception("no folder train-jpg to create training data")

if os.path.exists('data/train'):
    rmtree('data/train')

os.mkdir('data/train')
os.mkdir('data/train/valid')
os.mkdir('data/train/train')

files = []
images_by_class = images_by_classifications()
for classification in images_by_class.keys():
    images = images_by_class[classification]
    files += images

files = list(set(files))
train = files[:34816]
valid = files[-5632:]
for f in train:
    copyfile('data/train-jpg/'+f+'.jpg', 'data/train/train/'+f+'.jpg')

for f in valid:
    copyfile('data/train-jpg/'+f+'.jpg', 'data/train/valid/'+f+'.jpg')
