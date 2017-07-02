import os
from image_loader import images_by_classifications
from shutil import copyfile

train_files = []
validation_files = []
images_by_class = images_by_classifications()
for classification in images_by_class.keys():
    train_files += images_by_class[classification][:125]
    validation_files += images_by_class[classification][200:275]

train_files = list(set(train_files))
validation_files = list(set(validation_files))
for f in train_files:
    copyfile('train-jpg/'+f+'.jpg', 'sample/train/'+f+'.jpg')

for f in validation_files:
    copyfile('train-jpg/'+f+'.jpg', 'sample/valid/'+f+'.jpg')
