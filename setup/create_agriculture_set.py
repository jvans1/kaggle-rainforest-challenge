import os
import numpy as np
from image_loading.image_loader import images_by_classifications
from shutil import copyfile, rmtree

klass = "agriculture"
path = 'data/samples/'+klass
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

klass_images = images_by_class[klass][:1000]

num_for_validation = len(klass_images) / 5
random = np.random.permutation(klass_images)
train = random[num_for_validation:]
validation = random[:num_for_validation]
print("Training set has " +str(len(train)) + " images from the specific class")
print("Validation set has " +str(len(validation)) + " images from the specific class")
for f in train:
    copyfile('data/train-tif-v2/'+f+'.tif', path+'/train/'+f+'.tif')

for f in validation:
    copyfile('data/train-tif-v2/'+f+'.tif', path+'/valid/'+f+'.tif')

images_by_class.pop(klass)

train = np.asarray([])
valid = np.asarray([])
for key in  images_by_class:
    random = np.random.permutation(images_by_class[key])
    train = np.concatenate((train, random[:60]))
    valid = np.concatenate((valid, random[-20:]))

print("Adding " +str(len(train))+ " other types of images to training set")
print("Adding " +str(len(valid))+ " other types of images to valid set")

for f in train:
    copyfile('data/train-tif-v2/'+f+'.tif', path+'/train/'+f+'.tif')

for f in valid:
    copyfile('data/train-tif-v2/'+f+'.tif', path+'/valid/'+f+'.tif')
