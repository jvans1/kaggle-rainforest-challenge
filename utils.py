import csv
def binary_one_hot_mapping(klass):
    mapping = {}
    for fname in images_to_class:
        if klass in images_to_class[fname]:
            mapping[fname] = [0, 1]
        else:
            mapping[fname] = [1, 0]
    return mapping

def images_to_class_mapping(img_format="tif", filename="train_v2.csv"):
    images = {}
    with open(filename) as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            i += 1
            if i == 1: continue
            images[row[0]+"."+img_format] = row[1].split(' ')
    return images

images_to_class = images_to_class_mapping()
