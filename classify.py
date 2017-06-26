from image_loader import CSVLabelsToOneHot, DirectoryIterator
import pdb
classifications = [
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

labels = CSVLabelsToOneHot(classifications, 'train_v2.csv').generate_one_hot()

it = DirectoryIterator('sample/valid', labels)
a, b = next(it)
print(b[:2])
