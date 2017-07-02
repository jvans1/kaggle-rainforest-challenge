import os
from image_loading.image_loader import images_by_classifications
from shutil import copyfile, rmtree

if os.path.exists('sample'):
    rmtree('sample')
    os.mkdir('sample')
    os.mkdir('sample/valid')
    os.mkdir('sample/train')
else:
    os.mkdir('sample')
    os.mkdir('sample/valid')
    os.mkdir('sample/train')


train_files = []
validation_files = []
images_by_class = images_by_classifications()
train_files = list(set(train_files))
validation_files = list(set(validation_files))

for classification in images_by_class.keys():
    train_files += images_by_class[classification][:90]
    validation_files += images_by_class[classification][200:265]
for f in train_files:
    copyfile('train-jpg/'+f+'.jpg', 'sample/train/'+f+'.jpg')

for f in validation_files:
    copyfile('train-jpg/'+f+'.jpg', 'sample/valid/'+f+'.jpg')
