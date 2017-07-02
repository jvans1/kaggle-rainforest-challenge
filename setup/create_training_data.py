import os
from shutil import copyfile, rmtree
from importlib import import_module
from image_loading.image_loader import images_by_classifications

if not os.path.exists('train-jpg'):
    raise Exception("no folder train-jpg to create training data")

if os.path.exists('train'):
    rmtree('train')

os.mkdir('train')
os.mkdir('train/valid')
os.mkdir('train/train')

train_files = []
validation_files = []

images_by_class = images_by_classifications()
for classification in images_by_class.keys():
    images = images_by_class[classification]
    num_files_in_class = len(images)
    ten_percent =  num_files_in_class / 10
    train_files += images[ten_percent:]
    validation_files += images[:ten_percent]

tf = list(set(train_files))
vf = list(set(validation_files))
print("list of tf " + str(len(tf)))
for f in tf:
    copyfile('train-jpg/'+f+'.jpg', 'train/train/'+f+'.jpg')

for f in vf:
    copyfile('train-jpg/'+f+'.jpg', 'train/valid/'+f+'.jpg')

