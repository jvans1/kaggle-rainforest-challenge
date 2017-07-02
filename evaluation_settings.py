from keras.callbacks import ModelCheckpoint, BaseLogger, History, LearningRateScheduler
import numpy as np
import os
from image_loader import classify_images, DirectoryIterator, load_to_numpy, one_hot_to_labels
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


class TrainingSettings():
    def __init__(self, batch_size = 64):
        self.train_folder = 'train/train'
        self.validation_folder = 'train/valid'
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
        self.callbacks = [
            LearningRateScheduler(learning_rate_scheduler),
            History(),
            ModelCheckpoint("weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1),
            BaseLogger()
        ]


class SampleTrainingSettings():
    def __init__(self, batch_size = 16):
        self.train_folder = 'sample/train'
        self.validation_folder = 'sample/valid'
        self.batch_size = batch_size
        training_filenames = os.listdir(self.train_folder)
        validation_filenames = os.listdir(self.validation_folder)
        self.validation_filenames =  validation_filenames
        self.training_filenames =  training_filenames
        self.measure_classes = ['cloudy']
        self.measure_classes_indexes = [ CLASSES.index(classification) for classification in self.measure_classes]
        self.training_size = len(training_filenames)
        self.validation_size = len(validation_filenames)
        self.training_classes = CLASSES
        self.training_batch_count = self.training_size / self.batch_size + 1
        self.validation_batch_count = self.validation_size / self.batch_size + 1
        self.callbacks = [
            LearningRateScheduler(learning_rate_scheduler),
            History(),
            BaseLogger()
        ]


def learning_rate_scheduler(index):
    if index > 14:
        return 0.00001
    elif index >= 10:
        return 0.0001
    else:
        return 0.001

class EvaluationSettings():
    def __init__(self, batch_size = 16):
        self.train_folder = 'test-jpg'
        self.batch_size = batch_size
        training_filenames = os.listdir(self.train_folder)
        self.training_filenames =  training_filenames
        self.training_size = len(training_filenames)
        self.training_classes = CLASSES
        self.training_batch_count = self.training_size / self.batch_size + 1
        self.callbacks = []
