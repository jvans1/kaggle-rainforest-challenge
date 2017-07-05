from models.model import vgg16bn
from models.specialier import specializer
from config import CLASSES

specializers = []
vgg_model = vgg16bn(CLASSES, weights_file="")
for classification in CLASSES:
    specialized_model = specialier(classification, 0.0)
    specialized_model.load_weights("weights/"+classification+"-weights")
    specializers.append(specialized_model)


outputs = [  s_model(vgg_model.output) for s_model in  specializers]
model = Model(inputs=vgg_model.inputs, outputs=outputs)
