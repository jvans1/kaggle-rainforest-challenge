import os
from image_loader import images_by_classifications
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

if not os.path.exists('train'):
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
