from keras.callbacks import ModelCheckpoint, BaseLogger, History
import datetime
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
class PredictSettings():
    def __init__(self, folder,  batch_size=40, log_file="results", model_type=""):
        self.folder = folder
        self.batch_size = batch_size
        filenames = os.listdir(self.folder)
        self.filenames =  filenames
        self.training_size = len(filenames)
        self.training_classes = CLASSES
        self.batch_count = (self.training_size / self.batch_size)
        default_callbacks = [
            History(),
            BaseLogger()
        ]
        self.callbacks = default_callbacks




class Settings():
    def __init__(self, folder, batch_size=40, log_file="results", model_type=""):
        self.train_folder = folder + "/train"
        self.validation_folder = folder + "/valid"
        self.batch_size = batch_size
        training_filenames = os.listdir(self.train_folder)
        validation_filenames = os.listdir(self.validation_folder)
        self.validation_filenames =  validation_filenames
        self.training_filenames =  training_filenames
        self.training_size = len(training_filenames)
        self.validation_size = len(validation_filenames)
        self.training_classes = CLASSES
        self.training_batch_count = (self.training_size / self.batch_size) + 1
        self.validation_batch_count = (self.validation_size / self.batch_size) +1
        current_time = str(datetime.datetime.now())
        default_callbacks = [
            History(),
            CSVLogger(log_file+"--"+current_time+".csv", separator=',', append=False),
            ModelCheckpoint("weights/"+model_type+"-weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1),
            BaseLogger()
        ]
        self.callbacks = default_callbacks



