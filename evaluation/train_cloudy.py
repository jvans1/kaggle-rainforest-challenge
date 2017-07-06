from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import pdb
from keras.preprocessing.image import ImageDataGenerator
from config import Settings
from image_loading.image_loader import train_generators
from models.specializer import specializer
from models.linear_model import dense_model
from image_loading.utils import binary_one_hot_mapping

condition = "partly_cloudy"
folder = "data/samples/"+condition
settings = Settings(folder, model_type=condition)
img_gen = ImageDataGenerator(rotation_range=180, shear_range=0.3, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)
mapping = binary_one_hot_mapping(condition)

gen, val_gen = train_generators(settings, mapping=mapping, image_processor=img_gen)

model = dense_model()# specializer(condition, 0.0)
#model.load_weights('weights/'+condition+'final')

def learning_rate_scheduler(index):
    if index > 2:
        return 0.001
    else:
        return 0.01

learning_rate_annealer = LearningRateScheduler(learning_rate_scheduler)
callbacks = settings.callbacks + [learning_rate_annealer]

model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(gen, settings.training_batch_count, epochs=4, validation_data = val_gen, validation_steps=settings.validation_batch_count, callbacks=callbacks)
