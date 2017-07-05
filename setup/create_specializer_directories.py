from config import CLASSES
from image_loading.image_loader import images_by_classifications, image_label_mapping
import os


train_files = []
validation_files = []
images_by_class = images_by_classifications()
image_label_mapping = image_label_mapping()

for klass in CLASSES:
    if not os.path.exists('data/'+klass+):
        os.mkdir('data/'+klass)
        os.mkdir('data/'+klass+'/valid')
        os.mkdir('data/'+klass+'/train')

