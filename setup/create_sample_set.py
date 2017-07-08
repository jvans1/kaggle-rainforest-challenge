import os
from image_loading.image_loader import images_by_classifications, image_label_mapping
from shutil import copyfile, rmtree
import operator

if os.path.exists('data/sample'):
    rmtree('data/sample')
    os.mkdir('data/sample')
    os.mkdir('data/sample/valid')
    os.mkdir('data/sample/train')
else:
    os.mkdir('data/sample')
    os.mkdir('data/sample/valid')
    os.mkdir('data/sample/train')


train_files = []
validation_files = []
images_by_class = images_by_classifications()
image_label_mapping = image_label_mapping()

def count_labels(files, image_label_mapping):
    counts = {}
    for f in files:
        f = '_'.join(f.split('_')[-2:])
        for label in image_label_mapping[f]:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

    s= ""

    sorted_keys = sorted(counts.items(), key=operator.itemgetter(1))
    for key, count in sorted_keys:
        s+= key+" : " + str(count) + "\n"

    print(s)

for classification in images_by_class.keys():
    images_in_class =  images_by_class[classification][:90]
    train_files += images_in_class
    validation_files += images_by_class[classification][200:265]

train_files = list(set(train_files))
validation_files = list(set(validation_files))

for f in train_files:
    copyfile('data/train-tif-v2/'+f+'.tif', 'data/sample/train/'+f+'.tif')

for f in validation_files:
    copyfile('data/train-tif-v2/'+f+'.tif', 'data/sample/valid/'+f+'.tif')


count_labels(os.listdir('data/train-tif'), image_label_mapping)
