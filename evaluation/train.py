from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import pdb
from keras.preprocessing.image import ImageDataGenerator

klass = "partly_cloudy"
mapping = binary_one_hot_mapping(klass)

batch_size = 64

folder = "data/train/"
train_folder = folder+"train"
train_filenames = os.listdir(train_folder)
validation_folder = folder+"valid"
validation_filenames = os.listdir(validation_folder)
train_batch_count = len(train_filenames) / batch_size
val_batch_count = len(validation_filenames) / batch_size

gen_t = image.ImageDataGenerator(vertical_flip=True, horizontal_flip=True)

gen = IM.DirectoryIterator(train_folder, train_filenames, image_processor=gen_t, one_hot_filename_mapping=mapping, batch_size=batch_size, shuffle=True)
val_gen = IM.DirectoryIterator(validation_folder, validation_filenames, one_hot_filename_mapping=mapping, batch_size=batch_size, shuffle=False)

for l in full_model.layers:
    l.trainable = True

full_model.compile(optimizer=(Adam(lr=0.00001)), metrics=["accuracy"], loss="categorical_crossentropy")
preds = full_model.fit_generator(gen, train_batch_count, epochs=2, validation_data=val_gen, validation_steps=val_batch_count , callbacks=callbacks)

