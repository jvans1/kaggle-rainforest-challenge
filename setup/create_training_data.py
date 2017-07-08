import os
from shutil import copyfile, rmtree
from importlib import import_module
from image_loading.image_loader import images_by_classifications

if not os.path.exists('data/train-tif-v2'):
    raise Exception("no folder train-tif-v2 to create training data")

if os.path.exists('data/train'):
    rmtree('data/train')

os.mkdir('data/train')
os.mkdir('data/train/valid')
os.mkdir('data/train/train')

train_files = []
validation_files = []

images_by_class = images_by_classifications()
for classification in images_by_class.keys():
    images = images_by_class[classification]
    num_files_in_class = len(images)
    eighteen_percent =  num_files_in_class / 7
    train_files += images[eighteen_percent:]
    class_val_files = images[:eighteen_percent]
    validation_files += class_val_files
    print("creating " +str(len(class_val_files))+" validation files for class " +classification)

tf = list(set(train_files))
vf = list(set(validation_files))
print("list of validation files " + str(len(vf)))
for f in tf:
    copyfile('data/train-tif-v2/'+f+'.tif', 'data/train/train/'+f+'.tif')

for f in vf:
    copyfile('data/train-tif-v2/'+f+'.tif', 'data/train/valid/'+f+'.tif')
