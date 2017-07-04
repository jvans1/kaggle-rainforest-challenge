from keras.callbacks import ModelCheckpoint, BaseLogger, History, LearningRateScheduler
import datetime
import numpy as np
from keras.callbacks import CSVLogger
import os
CLASSES = [
    "agriculture",
    "artisinal_mine",
    "bare_ground",
    "blooming",
    "blow_down",
    "clear",
    "cloudy",
    "conventional_mine",
    "cultivation",
    "habitation",
    "haze",
    "partly_cloudy",
    "primary",
    "road",
    "selective_logging",
    "slash_burn",
    "water"
]

class SettingsWithValidation():
    def __init__(self, folder, validation_folder, batch_size=40, log_file="results", extra_callbacks=[]):
        self.train_folder = folder
        self.validation_folder = validation_folder
        self.batch_size = batch_size
        training_filenames = os.listdir(self.train_folder)
        validation_filenames = os.listdir(self.validation_folder)
        self.validation_filenames =  validation_filenames
        self.training_filenames =  training_filenames
        self.training_size = len(training_filenames)
        self.validation_size = len(validation_filenames)
        self.training_classes = CLASSES
        self.training_batch_count = self.training_size / self.batch_size + 1
        self.validation_batch_count = self.validation_size / self.batch_size + 1
        current_time = str(datetime.datetime.now())
        default_callbacks = [
            LearningRateScheduler(learning_rate_scheduler),
            History(),
            CSVLogger(log_file+"--"+current_time+".csv", separator=',', append=False),
            ModelCheckpoint("weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1),
            BaseLogger()
        ]
        self.callbacks = default_callbacks + extra_callbacks



def learning_rate_scheduler(index):
    if index > 12:
        return 0.0000001
    elif index > 8:
        return 0.000001
    elif index > 5:
        return 0.00001
    else:
        return 0.0001
