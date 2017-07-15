import os
import numpy as np
import pdb
from image_loading.image_loader import images_by_classifications
from shutil import copyfile, rmtree

path = 'data/rare'
if os.path.exists(path):
    rmtree(path)
    os.mkdir(path)
    os.mkdir(path+'/valid')
    os.mkdir(path+'/train')
else:
    os.mkdir(path)
    os.mkdir(path+'/valid')
    os.mkdir(path+'/train')


#    'artisinal_mine': 339,
#    'bare_ground': 862,
#    'blow_down': 101,
#    'blooming': 332,
#    'conventional_mine': 100,
#    'selective_logging': 340,
#    'slash_burn': 209,

#rare   total: 2283
#others total:  100 * 6 = 600
#total       : 2883
#ratios = [   ]
images_by_class = images_by_classifications()


rare_images_train = []
rare_images_valid = []
imgs =[
 images_by_class["selective_logging"],
 images_by_class["conventional_mine"],
 images_by_class["slash_burn"],
 images_by_class["artisinal_mine"],
 images_by_class["blooming"],
 images_by_class["bare_ground"],
 images_by_class["blow_down"]
]

def diff(imgs, others):
    return list(set(others) - set(imgs))

others = [
 images_by_class["cultivation"],
 images_by_class["habitation"],
 images_by_class["primary"],
 images_by_class["water"],
 images_by_class["agriculture"],
 images_by_class["road"]
]

#    GT 1000
#    'cultivation': 4547,
#    'habitation': 3660,
#    'primary': 37513,
#    'water': 7411,
#    'agriculture': 12315,
#    'road': 8071}

counts = { }
rare = ["selective_logging","conventional_mine", "slash_burn", "artisinal_mine", "blooming", "bare_ground", "blow_down" ]
common = ['cultivation', 'habitation', 'primary', 'water', 'agriculture', 'road']
for img_set, label in zip(imgs + others, rare + common ):

    rand = np.random.permutation(img_set)
    alll = np.concatenate((rare_images_train, rare_images_valid))
    eligible = diff(alll, rand)
    if label in common:
        eligible = eligible[:100]
    print(label + " Count: " +str(len(eligible)))
    cutoff = len(eligible) / 5
    train = eligible[cutoff:]
    valid = eligible[:cutoff]
    counts[label] = len(train)
    print("            " +str(len(train)) + " " + label + " to training set")
    print("            " +str(len(valid)) + " " + label + " to validation set")
    rare_images_train = np.concatenate((rare_images_train, train))
    rare_images_valid = np.concatenate((rare_images_valid, valid))

rare_images_train = list(set(rare_images_train))
rare_images_valid = list(set(rare_images_valid))
print("Putting " +str(len(rare_images_train)) + " rare files in training set")
print("Putting " +str(len(rare_images_valid)) + " rare files in validation set")

weights = {}

total = len(rare_images_train)
for k in rare:
    weights[k] = total / counts[k]

print("Weights are " + str(weights))
for f in rare_images_train:
    copyfile('data/train-jpg/'+f+'.jpg', path+'/train/'+f+'.jpg')

for f in rare_images_valid:
    copyfile('data/train-jpg/'+f+'.jpg', path+'/valid/'+f+'.jpg')
