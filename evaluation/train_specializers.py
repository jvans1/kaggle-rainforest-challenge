from config import Settings
from config import CLASSES

for classification in CLASSES:
    train_folder = "data/"+classification+"/train"
    validation_folder = "data/"+classification+"/valid"
    settings = Settings(train_folder, validation_folder)
    gen         = train_generator(settings)
    val_gen     = train_generator(settings)

    model = specializer(classification)
    model.fit_generator(gen, settings.training_batch_count, epochs=15, validation_data = val_gen, validation_steps=settings.validation_batch_count, callbacks=settings.callbacks)
